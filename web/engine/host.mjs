// Runs the engine module as an ordinary UCI engine on stdin/stdout, under Node.
//
// It exists so the browser artefact can be checked against the native binary with the
// same commands and a plain `diff`, without a browser in the loop.
//
//   node web/engine/host.mjs <path-to.wasm> [--net <net.bin>] [--tb <tb34.gtpk>] [--mem]  < session.txt
//
// `--net` installs weights the module was not built with — which is how the browser
// receives them, and therefore the path worth exercising here. `--tb` hands over the
// endgame tables the same way.
//
// `--mem` reports the linear memory high-water mark on stderr when the session ends,
// which is how the stack-size link argument gets chosen rather than guessed.

import { readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { loadEngine } from "./engine.mjs";

const wasmPath = process.argv[2];
const reportMemory = process.argv.includes("--mem");
if (!wasmPath) {
  console.error("usage: node host.mjs <path-to.wasm> [--net <net.bin>] [--tb <tb34.gtpk>] [--mem]");
  process.exit(2);
}

const engine = await loadEngine(readFileSync(wasmPath), (line) => {
  process.stdout.write(line + "\n");
});

const netFlag = process.argv.indexOf("--net");
if (netFlag !== -1) {
  const file = readFileSync(process.argv[netFlag + 1]);
  // Node hands back a Buffer over a shared pool; the slice is the bytes of this file
  // alone, which is what the module expects to size against.
  const bytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  if (!engine.loadNetwork(bytes)) {
    process.stderr.write("the engine refused the network\n");
    process.exit(1);
  }
}

const tbFlag = process.argv.indexOf("--tb");
if (tbFlag !== -1) {
  const file = readFileSync(process.argv[tbFlag + 1]);
  const bytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  if (!engine.loadTables(bytes)) {
    process.stderr.write("the engine refused the endgame tables\n");
    process.exit(1);
  }
}

const done = () => {
  if (reportMemory) {
    const mb = (engine.memoryBytes() / (1024 * 1024)).toFixed(1);
    process.stderr.write(`linear memory: ${mb} MB\n`);
  }
  process.exit(0);
};

const input = createInterface({ input: process.stdin, terminal: false });
for await (const line of input) {
  if (!engine.command(line)) break;
}
done();
