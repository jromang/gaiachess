// Measures whether a running search can actually be cut short.
//
// This is the one thing in the browser build that fails silently and invisibly: without
// cross-origin isolation the flag never reaches the worker, every search runs to its end,
// and the only symptom is an interface that feels sluggish when a game is taken back. So
// it is measured rather than assumed.
//
//   node tools/web/serve.mjs 8081 web/dist &     # sets COOP/COEP
//   python3 -m http.server 8080 --directory web/dist &   # does not
//   node tools/web/stop-test.mjs http://localhost:8081/stoptest.html
//
// Isolated, the search should end within a few milliseconds of the flag. Not isolated, it
// runs its full budget — which is correct, and what the interface already copes with.

import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://localhost:8081/stoptest.html";

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new",
  args: ["--no-sandbox", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("  [pageerror] " + e.message));

await page.goto(url, { waitUntil: "load", timeout: 60000 });
await page.waitForFunction("window.__done === true", { timeout: 120000 }).catch(() => {});

const report = await page.$eval("#out", (el) => el.textContent);
console.log(report.split(String.fromCharCode(10)).map((l) => "  " + l).join(String.fromCharCode(10)));
await browser.close();
