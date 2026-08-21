// Loader for the engine module.
//
// Deliberately hand-written and dependency-free: the interface module is loaded by
// miniquad's own JavaScript, which builds its import object itself and knows nothing of
// wasm-bindgen, so the engine is loaded the same way rather than dragging a second,
// incompatible toolchain into the page.
//
// The one rule worth remembering: any call into the module may grow the linear memory,
// which detaches every ArrayBuffer view onto it. Views are therefore built fresh at each
// use and never kept.

export async function loadEngine(wasmSource, onLine, stopFlag) {
  let memory;
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const imports = {
    env: {
      gaia_out(ptr, len) {
        onLine(decoder.decode(new Uint8Array(memory.buffer, ptr, len)));
      },
      // wasm32-unknown-unknown has no clock: std::time::Instant panics there, so the
      // engine reads the time through here instead. Monotonic is what matters, not the
      // origin — time management only ever looks at differences.
      gaia_now_ms() {
        return performance.now();
      },
      // Read once every 512 nodes from inside the search. A message could not do this:
      // a worker busy in a search reads nothing until it comes back out, so shared
      // memory is the only thing that crosses. Where the page is not cross-origin
      // isolated the buffer is an ordinary one, the flag never changes, and the search
      // simply runs to its end.
      gaia_host_stop() {
        return stopFlag ? Atomics.load(stopFlag, 0) : 0;
      },
    },
  };

  const { instance } =
    wasmSource instanceof Response || wasmSource instanceof Promise
      ? await WebAssembly.instantiateStreaming(wasmSource, imports)
      : await WebAssembly.instantiate(wasmSource, imports);

  const exports = instance.exports;
  memory = exports.memory;
  exports.gaia_new();

  return {
    /// Runs one UCI command. Returns false once the engine has been told to quit.
    command(line) {
      const bytes = encoder.encode(line);
      const ptr = exports.gaia_alloc(bytes.length);
      new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
      return exports.gaia_command(bytes.length) !== 0;
    },
    /// Installs the network weights.
    ///
    /// The bytes are written straight into the module's own copy rather than staged and
    /// copied: a browser never gets linear memory back, so a doubled peak is a doubled
    /// cost for the life of the page. Reserving may grow the memory, so the view is
    /// built after the call and never before.
    loadNetwork(bytes) {
      const size = exports.gaia_net_size();
      if (bytes.byteLength !== size) {
        throw new Error(`network is ${bytes.byteLength} bytes, expected ${size}`);
      }
      const ptr = exports.gaia_net_reserve();
      new Uint8Array(memory.buffer, ptr, size).set(new Uint8Array(bytes));
      return exports.gaia_net_finish() !== 0;
    },
    /// Bytes of linear memory currently reserved, for the record.
    memoryBytes() {
      return memory.buffer.byteLength;
    },
  };
}
