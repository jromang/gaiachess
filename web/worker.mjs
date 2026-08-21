// The engine, in a Web Worker.
//
// Everything the page asks for is answered here, so a search never touches the thread
// that draws. The worker is strictly sequential — it runs one command to completion
// before reading the next — which is what lets it stamp each answer with the generation
// of the `go` it just ran. The page therefore needs no bookkeeping of its own to work out
// whose answer is whose.
//
// Note what cannot be done here: a `stop` posted while a search runs will not be read
// until that search returns, because nothing interrupts a busy worker. The interface
// abandons results rather than stopping them, which it can afford because every rung
// below full strength is bounded by depth and comes back in milliseconds.

import { loadEngine } from "./engine.mjs";

let engine = null;
/** Commands that arrived before the module was ready. */
const waiting = [];
/** The move named by the last `bestmove` line, reset before each search. */
let lastBest = null;

const post = (message) => self.postMessage(message);

async function boot(wasmUrl, stopFlag) {
  engine = await loadEngine(fetch(wasmUrl), (line) => {
    if (line.startsWith("bestmove ")) {
      lastBest = line.split(" ")[1];
    }
    // Everything else is `info`, forwarded so a page that wants to show a depth or a
    // score can, and ignored otherwise.
    post({ k: "line", line });
  }, stopFlag);
  post({ k: "ready" });
  for (const message of waiting.splice(0)) handle(message);
}

function handle(message) {
  switch (message.k) {
    case "net": {
      let ok = false;
      let error = null;
      try {
        ok = engine.loadNetwork(message.bytes);
      } catch (err) {
        error = String(err);
      }
      post({ k: "net", ok, error, memory: engine.memoryBytes() });
      break;
    }

    case "cmd":
      engine.command(message.line);
      break;

    case "go":
      lastBest = null;
      engine.command(message.line);
      // `bestmove` is emitted during the call above, so by here it is known. A search
      // that named nothing means the engine found no move, which the interface reads as
      // the null move rather than as silence.
      post({ k: "best", gen: message.gen, uci: lastBest ?? "0000" });
      break;
  }
}

self.onmessage = (event) => {
  const message = event.data;
  if (message.k === "boot") {
    boot(message.wasm, message.stop);
  } else if (engine) {
    handle(message);
  } else {
    waiting.push(message);
  }
};
