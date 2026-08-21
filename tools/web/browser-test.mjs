// Plays a move in a real browser and photographs the result.
//
// The interface and the engine are two wasm modules talking through hand-written
// JavaScript; nothing about that chain is exercised by `cargo test`, and a broken import
// shows up as a stub that silently does nothing rather than as an error. So this drives
// an actual Chrome: it starts a game, drags a pawn, waits for the engine, and saves a
// screenshot at each step, failing loudly on any console error.
//
//   python3 -m http.server 8080 --directory web/dist &
//   npm install puppeteer-core        # drives the Chrome already installed; no download
//   node tools/web/browser-test.mjs http://localhost:8080/ /tmp/gaia
//
// The last argument is a prefix for the four screenshots. Set CHROME to point somewhere
// other than the default Windows install.

import puppeteer from "puppeteer-core";
const [url, prefix] = process.argv.slice(2);

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new",
  args: ["--no-sandbox", "--enable-unsafe-swiftshader", "--use-gl=angle",
         "--window-size=704,840", "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
await page.setViewport({ width: 704, height: 840 });
const log = [];
page.on("console", (m) => log.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => log.push(`[pageerror] ${e.message}`));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

await page.goto(url, { waitUntil: "load", timeout: 60000 });
await wait(12000);

// Demarrer la partie au clavier (aucune souris n'a encore bouge).
await page.keyboard.press("KeyX");
await wait(1500);
await page.screenshot({ path: `${prefix}-1-board.png` });

// e2 -> e4, en glisser-deposer. Geometrie relevee sur la capture : cases de 80x84,
// coin bas-gauche de l'echiquier vers (32, 744).
const sq = (file, rank) => [32 + file * 80 + 40, 744 - rank * 84 - 42];
const [fx, fy] = sq(4, 1);
const [tx, ty] = sq(4, 3);

await page.mouse.move(fx, fy);
await wait(400);
await page.mouse.down();
await wait(300);
await page.screenshot({ path: `${prefix}-2-grabbed.png` });
for (let i = 1; i <= 8; i++) {
  await page.mouse.move(fx + ((tx - fx) * i) / 8, fy + ((ty - fy) * i) / 8);
  await wait(60);
}
await wait(300);
await page.mouse.up();
await wait(1200);
await page.screenshot({ path: `${prefix}-3-played.png` });

// Laisser le moteur repondre.
await wait(8000);
await page.screenshot({ path: `${prefix}-4-reply.png` });

const lines = [...new Set(log)];
for (const line of lines) console.log("  " + line);
await browser.close();
// A clean console is the point: every fault in this chain is silent otherwise. `info`
// is exempt — the page uses it to say what it is doing, such as noting that it is not
// cross-origin isolated and therefore cannot cut a search short.
const faults = lines.filter((l) => !l.startsWith("[info]"));
process.exit(faults.length === 0 ? 0 : 1);
