// A static server for web/dist that sets the two headers a browser wants before it will
// hand a page a SharedArrayBuffer — the same pair itch.io sets behind its
// "SharedArrayBuffer support" box.
//
// `python3 -m http.server` is not enough for that, and without it the immediate-stop
// path cannot be tried at all: the page falls back to disowning searches instead of
// cutting them short, which is a different code path and hides any fault in this one.
//
//   node tools/web/serve.mjs [port] [dir]

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";

const port = Number(process.argv[2] ?? 8081);
const root = resolve(process.argv[3] ?? "web/dist");

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream",
  ".png": "image/png",
};

createServer(async (req, res) => {
  let name = decodeURI(req.url.split("?")[0]);
  if (name.endsWith("/")) name += "index.html";

  // Resolved and then checked to be inside the root, rather than trusting the request to
  // contain no way out of it.
  const file = resolve(join(root, name));
  if (file !== root && !file.startsWith(root + sep)) {
    res.writeHead(403).end("outside the served directory");
    return;
  }

  try {
    const body = await readFile(file);
    res.writeHead(200, {
      "content-type": types[extname(file)] ?? "application/octet-stream",
      "content-length": body.length,
      "cross-origin-opener-policy": "same-origin",
      "cross-origin-embedder-policy": "require-corp",
    });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, () => {
  console.log(`serving ${root} on http://localhost:${port}/ (cross-origin isolated)`);
});
