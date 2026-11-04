// Records the UI scenes of the demo video by driving the real app in Chromium.
//   node video/record.mjs <playwright-core dir> <chromium executable> [scene ...]
// Writes video/out/ui/<scene>.webm and video/out/ui/marks.json (action times in seconds from scene start).
import { createRequire } from "module";
import fs from "fs";
import path from "path";

const [pwDir, chrome, ...only] = process.argv.slice(2);
const { chromium } = createRequire(import.meta.url)(pwDir);
const OUT = path.resolve("video/out/ui");
const BASE = "http://localhost:8811";
const REAL = "0fbcad05f59c", RENDERED = "e7584905616b";
const VIEW = { width: 1920, height: 1080 };
fs.mkdirSync(OUT, { recursive: true });
const marksFile = path.join(OUT, "marks.json");
const marks = fs.existsSync(marksFile) ? JSON.parse(fs.readFileSync(marksFile, "utf8")) : {};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const phase = (page) => page.evaluate(() => phase);
async function waitPhase(page, want, timeout = 30000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (want.includes(await phase(page))) return; await sleep(50); }
  throw new Error(`timed out waiting for phase ${want}`);
}

async function scene(name, fn, launchArgs = []) {
  if (only.length && !only.includes(name)) return;
  const browser = await chromium.launch({ executablePath: chrome, args: launchArgs });
  const ctx = await browser.newContext({ viewport: VIEW, recordVideo: { dir: OUT, size: VIEW },
                                         permissions: ["microphone"] });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => console.error(`${name} page error:`, e.message));
  page.on("console", (msg) => msg.type() === "error" && console.error(`${name} console:`, msg.text()));
  const t0 = Date.now();
  const m = { _boxes: {} };
  const mark = (k) => { m[k] = (Date.now() - t0) / 1000; };
  await fn(page, mark, m);
  mark("end");
  const video = page.video();
  await ctx.close(); await browser.close();
  fs.renameSync(await video.path(), path.join(OUT, `${name}.webm`));
  marks[name] = m;
  fs.writeFileSync(marksFile, JSON.stringify(marks, null, 1));
  console.log(name, JSON.stringify(m));
}

async function openClip(page, mark, id) {
  await page.goto(`${BASE}/?clip=${id}`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => result !== null);
}

const box = async (page, sel) => page.locator(sel).boundingBox();

// Hero: the real clip, prediction freezes, arc draws, ball comes down on the mark. Twice.
await scene("hero", async (page, mark, m) => {
  await openClip(page, mark, REAL);
  mark("loaded");
  await waitPhase(page, ["frozen"]); mark("freeze1");
  await waitPhase(page, ["done"]); mark("reveal1");
  await sleep(3500);
  await page.click("#replayBtn"); mark("replay");
  await waitPhase(page, ["frozen"]); mark("freeze2");
  await waitPhase(page, ["done"]); mark("reveal2");
  await sleep(3000);
  m._boxes.stage = await box(page, "#stage");
  m._boxes.notes = await box(page, "aside.notes");
});

// Second take: the rendered 240 fps throw, played back in slow motion.
await scene("second", async (page, mark, m) => {
  await openClip(page, mark, RENDERED);
  mark("loaded");
  await waitPhase(page, ["frozen"], 60000); mark("freeze");
  await waitPhase(page, ["done"], 60000); mark("reveal");
  await sleep(3500);
  m._boxes.stage = await box(page, "#stage");
});

// What-if: hold the mic (fed a spoken question), Moon ghost, then "Twice as hard".
const askWav = path.resolve("video/out/voice/ask_padded.wav");
await scene("whatif", async (page, mark, m) => {
  await openClip(page, mark, REAL);
  await waitPhase(page, ["done"]); mark("ready");
  await sleep(1200);
  const mic = page.locator("#mic");
  await mic.hover();
  await page.mouse.down(); mark("mic_down");
  await sleep(2600);
  await page.mouse.up(); mark("mic_up");
  await waitPhase(page, ["ghost"], 60000); mark("ghost1");
  await sleep(6500);
  await page.click('button.quick[data-q="What if I threw it twice as hard?"]'); mark("twice_click");
  await page.waitForFunction(() => /twice/i.test(document.getElementById("ghostChip").textContent), null, { timeout: 60000 });
  mark("ghost2");
  await sleep(6000);
  m.caption = await page.locator("#caption").innerText();
  m._boxes.stage = await box(page, "#stage");
  m._boxes.askbar = await box(page, "#whatif");
}, ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${askWav}%noloop`]);

// Animated scenes from video/scenes.html, timed to the narration cues in video/out/plan.json.
const planFile = path.resolve("video/out/plan.json");
const readResult = (id) => JSON.parse(fs.readFileSync(`cache/clips/${id}/result.json`, "utf8"));
for (const [vo, id] of [["title", "title"], ["pipeline", "pipe"], ["reasoning", "agent"], ["outro", "outro"]]) {
  if (only.length && !only.includes(id)) continue;
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8"))[vo];
  const data = { scene: id, cues: plan.cues, end: plan.end, real: readResult(REAL), rendered: readResult(RENDERED),
                 media: `${BASE}/media/${REAL}/` };
  await scene(id, async (page, mark) => {
    await page.addInitScript((d) => { window.DATA = d; }, data);
    await page.goto("file://" + path.resolve("video/scenes.html"));
    await page.evaluate(() => window.READY);
    await sleep(300);
    mark("start");
    await page.evaluate(() => window.start());
    await sleep(plan.dur * 1000 + 300);
  });
}
