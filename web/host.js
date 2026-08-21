// What the interface module talks to.
//
// A classic script, not a module, because miniquad's gl.js is one and puts its hooks
// (`miniquad_add_plugin`, `load`, `wasm_memory`, `UTF8ToString`) on the global object.
// It registers the seven functions the interface imports, owns the Web Worker the engine
// runs in, fetches the weights, tells the interface which languages the reader asks for,
// and turns the synthesised clips into Web Audio.
//
// Nothing here is generated. The interface is brought up by gl.js, which builds its own
// import object and knows nothing of wasm-bindgen; adding a second, incompatible loader
// to the page to gain a clock and an audio call would be a poor trade.

(function () {
  "use strict";

  // ---------------------------------------------------------------- engine

  const worker = new Worker("worker.mjs", { type: "module" });

  /** Answers waiting to be collected by the interface, oldest first. */
  const answers = [];
  /** `go` lines held back until the weights are in. */
  const heldSearches = [];
  let networkReady = false;
  /** Percent of the weights downloaded, read by the interface once a frame. */
  let netPercent = 0;

  // Raised to ask a running search to stop. It has to be shared memory: a worker in the
  // middle of a search reads no messages until it comes back out, so nothing else
  // crosses. `SharedArrayBuffer` needs the page to be cross-origin isolated — the
  // "SharedArrayBuffer support" box on itch, COOP/COEP anywhere else. Without it the
  // buffer is an ordinary one, the worker gets a copy, and the flag simply never
  // arrives: searches run to their end and their answers are dropped on arrival, which
  // is what every rung below full strength did anyway.
  const shared = typeof SharedArrayBuffer !== "undefined" && self.crossOriginIsolated;
  const stopFlag = new Int32Array(
    shared ? new SharedArrayBuffer(4) : new ArrayBuffer(4),
  );
  if (!shared) {
    console.info(
      "not cross-origin isolated: a search cannot be cut short, only disowned",
    );
  }

  worker.onmessage = (event) => {
    const message = event.data;
    switch (message.k) {
      case "ready":
        loadNetwork();
        break;
      case "net":
        networkReady = message.ok;
        if (!message.ok) {
          fail("the engine could not read the network: " + (message.error ?? "unknown"));
          return;
        }
        netPercent = 100;
        setStatus(null);
        // Anything the player set going while the weights were still arriving runs now.
        for (const held of heldSearches.splice(0)) worker.postMessage(held);
        break;
      case "best":
        answers.push(message.gen + " " + message.uci);
        break;
      case "line":
        // `info` lines. Nothing on screen shows them; kept for the console.
        break;
    }
  };

  worker.postMessage({ k: "boot", wasm: "engine.wasm", stop: stopFlag });

  async function loadNetwork() {
    if (typeof DecompressionStream === "undefined") {
      fail("this browser cannot decompress the engine weights (DecompressionStream)");
      return;
    }
    try {
      const response = await fetch("net.bin.deflate");
      if (!response.ok) throw new Error("HTTP " + response.status);

      // Progress is counted on the compressed stream, which is what `content-length`
      // describes and what the wait is actually made of. Reported rather than merely
      // awaited: this is tens of megabytes, and a page that looks frozen is
      // indistinguishable from one that is.
      const total = Number(response.headers.get("content-length")) || 0;
      let received = 0;
      const counted = new TransformStream({
        transform(chunk, controller) {
          received += chunk.length;
          // Reported to the interface rather than written into the page: it draws the
          // bar itself, in its own pixels at its own scale, so the wait looks like part
          // of the game instead of a caption stuck underneath it.
          if (total) netPercent = Math.round((received / total) * 100);
          controller.enqueue(chunk);
        },
      });

      // Decompressed by the browser itself: native code, and none of it in the module's
      // linear memory, which is never handed back once taken. Raw deflate rather than
      // gzip, so that no host takes the file for one it should unwrap in transport and
      // leaves the page with something it cannot read — see tools/web/build.sh.
      const stream = response.body
        .pipeThrough(counted)
        .pipeThrough(new DecompressionStream("deflate-raw"));
      const bytes = await new Response(stream).arrayBuffer();

      // Transferred, not copied: the page has no further use for it.
      worker.postMessage({ k: "net", bytes }, [bytes]);
    } catch (err) {
      fail("the engine weights could not be loaded: " + err);
    }
  }

  // ---------------------------------------------------------------- sound

  let audio = null;
  const clips = [];

  function audioContext() {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    return audio;
  }

  // A browser will not let a page make a noise before it has been touched, so the
  // context is woken by the first gesture and never asked again.
  for (const event of ["pointerdown", "keydown", "touchstart"]) {
    window.addEventListener(event, () => audioContext().resume(), { once: true });
  }

  // ---------------------------------------------------------------- status

  function setStatus(text) {
    const el = document.getElementById("status");
    if (!el) return;
    el.textContent = text ?? "";
    el.style.display = text ? "block" : "none";
  }

  function fail(text) {
    setStatus(text);
    const el = document.getElementById("status");
    if (el) el.classList.add("failed");
    console.error(text);
  }

  // ---------------------------------------------------------------- plugin

  miniquad_add_plugin({
    // Named and versioned so gl.js compares this against `gaia_crate_version` in the
    // module: the imports below are a contract written by hand on both sides, and a
    // drift would otherwise surface as a stub that silently does nothing.
    name: "gaia",
    version: 2,
    register_plugin(importObject) {
      const env = importObject.env;

      // wasm32-unknown-unknown has no clock of its own; std::time::Instant panics there.
      // Only differences are ever read, so any monotonic origin will do.
      env.gaia_now_ms = () => performance.now();

      env.gaia_engine_send = (ptr, len) => {
        worker.postMessage({ k: "cmd", line: UTF8ToString(ptr, len) });
      };

      env.gaia_engine_go = (ptr, len, generation) => {
        // Lowered here rather than when the answer comes back: this runs on the page,
        // before the worker is told to start, so the search never sees a flag left
        // standing from the last one.
        Atomics.store(stopFlag, 0, 0);
        const message = { k: "go", line: UTF8ToString(ptr, len), gen: generation };
        // Held rather than dropped: a game begun before the weights arrive should wait,
        // not be answered by an engine with no evaluation to speak of.
        if (networkReady) worker.postMessage(message);
        else heldSearches.push(message);
      };

      env.gaia_engine_poll = (buf, cap) => {
        if (answers.length === 0) return -1;
        const bytes = new TextEncoder().encode(answers[0]);
        if (bytes.length > cap) {
          answers.shift();
          return -1;
        }
        answers.shift();
        new Uint8Array(wasm_memory.buffer, buf, bytes.length).set(bytes);
        return bytes.length;
      };

      env.gaia_engine_abort = () => {
        Atomics.store(stopFlag, 0, 1);
      };

      // The interface module carries the search code without ever running it, so it
      // imports this too. It has no search to stop.
      env.gaia_host_stop = () => 0;

      env.gaia_net_progress = () => netPercent;

      // The languages the reader asks for, best first, as a comma-separated list of
      // BCP 47 tags. `?lang=` comes ahead of them: it is how the itch embed, and anyone
      // testing, names a language outright. Read once at start-up — a browser does not
      // change its mind mid-session, and the interface has a language row for whoever
      // this guesses wrong about.
      env.gaia_locale = (buf, cap) => {
        const asked = new URLSearchParams(location.search).get("lang");
        const preferred = navigator.languages || [navigator.language];
        const tags = (asked ? [asked, ...preferred] : [...preferred]).filter(Boolean);
        let list = tags.join(",");
        let bytes = new TextEncoder().encode(list);
        // Cut to whole tags rather than sending half of one. The tail is the part least
        // likely to have been answered anyway.
        while (bytes.length > cap && list.includes(",")) {
          list = list.slice(0, list.lastIndexOf(","));
          bytes = new TextEncoder().encode(list);
        }
        if (bytes.length === 0 || bytes.length > cap) return -1;
        new Uint8Array(wasm_memory.buffer, buf, bytes.length).set(bytes);
        return bytes.length;
      };

      env.gaia_sfx_register = (id, ptr, len, rate) => {
        const context = audioContext();
        const buffer = context.createBuffer(1, len, rate);
        // Copied out at once: the samples live in linear memory, which any later call
        // into the module may move.
        buffer.copyToChannel(new Float32Array(wasm_memory.buffer, ptr, len).slice(), 0);
        clips[id] = buffer;
      };

      env.gaia_sfx_play = (id, gain) => {
        const buffer = clips[id];
        if (!buffer || !audio || audio.state !== "running") return;
        const source = audio.createBufferSource();
        source.buffer = buffer;
        const volume = audio.createGain();
        volume.gain.value = gain;
        source.connect(volume).connect(audio.destination);
        // Started at once rather than on a buffer boundary — which is the whole reason
        // the desktop build went looking past the window library's own mixer.
        source.start();
      };
    },
  });

  // No caption: the interface draws its own progress. This is for failures only.
  load("gui.wasm");
})();
