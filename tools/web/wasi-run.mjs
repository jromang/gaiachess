// Cargo runner for the wasm32-wasip1 unit tests, which hold the simd128 kernels against
// the scalar build of the same module:
//
//   rustup target add wasm32-wasip1
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings tools/web/wasi-run.mjs" \
//   RUSTFLAGS="-C target-feature=+simd128" \
//     cargo test --lib --no-default-features --target wasm32-wasip1 -- horizontal_sum dense_layers
//
// stdout and stderr are written from JavaScript rather than through node's own fd_write.
// That one allocates through uvwasi, which reports the buffer to V8 as external memory;
// with a large linear memory (a whole network is 87 MB) the report triggers a full GC
// from inside the call, and the GC's stack walk crashes on the wasm frame beneath it
// (node 22, SIGSEGV in InnerPointerToCodeCache::GetCacheEntry).
import { readFile } from "node:fs/promises";
import { writeSync } from "node:fs";
import { WASI } from "node:wasi";

const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [file, ...args], env: process.env, preopens: { "/": "/" } });
const imports = wasi.getImportObject();
const native = imports.wasi_snapshot_preview1;
let memory;

const fd_write = (fd, iovs, iovsLen, nwritten) => {
  if (fd !== 1 && fd !== 2) return native.fd_write(fd, iovs, iovsLen, nwritten);
  const view = new DataView(memory.buffer);
  let total = 0;
  for (let i = 0; i < iovsLen; i++) {
    const ptr = view.getUint32(iovs + 8 * i, true);
    const len = view.getUint32(iovs + 8 * i + 4, true);
    let off = 0;
    while (off < len) off += writeSync(fd, new Uint8Array(memory.buffer, ptr + off, len - off));
    total += len;
  }
  view.setUint32(nwritten, total, true);
  return 0; // ESUCCESS
};

imports.wasi_snapshot_preview1 = { ...native, fd_write };
const mod = await WebAssembly.compile(await readFile(file));
const inst = await WebAssembly.instantiate(mod, imports);
memory = inst.exports.memory;
process.exitCode = wasi.start(inst);
