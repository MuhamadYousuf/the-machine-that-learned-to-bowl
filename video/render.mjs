// Batch renderer: bundles the Remotion project once, then renders jobs from a JSON file.
//
//   node render.mjs jobs/previews.json            # render every job
//   node render.mjs jobs/footage.json --skip-done # resume: skip outputs that already exist
//
// A job is {id, composition, props, out, still?: frameNumber | frameNumber[]}.
// Stills render PNGs (one per listed frame); other jobs render H.264 MP4s.
import {bundle} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [jobFile, ...flags] = process.argv.slice(2);
if (!jobFile) {
  console.error('usage: node render.mjs <jobs.json> [--skip-done] [--only id1,id2]');
  process.exit(1);
}
const skipDone = flags.includes('--skip-done');
const onlyFlag = flags.find((f) => f.startsWith('--only='));
const only = onlyFlag ? new Set(onlyFlag.slice(7).split(',')) : null;
const jobs = JSON.parse(fs.readFileSync(jobFile, 'utf8')).filter((j) => !only || only.has(j.id));

const chromiumOptions = {gl: 'angle'};
console.log(`bundling...`);
const serveUrl = await bundle({entryPoint: path.join(here, 'src', 'index.ts')});
const log = [];
for (const job of jobs) {
  const out = path.resolve(here, job.out);
  fs.mkdirSync(path.dirname(out), {recursive: true});
  const frames = job.still === undefined ? null : [].concat(job.still);
  const targets = frames ? frames.map((f) => out.replace(/\.png$/, frames.length > 1 ? `_${f}.png` : '.png')) : [out];
  if (skipDone && targets.every((t) => fs.existsSync(t))) {
    console.log(`skip ${job.id}`);
    continue;
  }
  const started = Date.now();
  const composition = await selectComposition({serveUrl, id: job.composition, inputProps: job.props ?? {}, chromiumOptions});
  if (frames) {
    for (let i = 0; i < frames.length; i++) {
      await renderStill({composition, serveUrl, output: targets[i], frame: frames[i], inputProps: job.props ?? {}, chromiumOptions});
    }
  } else {
    let last = -1;
    try {
    await renderMedia({
      composition,
      serveUrl,
      codec: 'h264',
      crf: 16,
      pixelFormat: 'yuv420p',
      x264Preset: 'slow',
      imageFormat: 'png',
      colorSpace: 'bt709',
      outputLocation: out,
      inputProps: job.props ?? {},
      chromiumOptions,
      concurrency: job.concurrency ?? 6,
      onProgress: ({progress}) => {
        const pct = Math.floor(progress * 10);
        if (pct !== last) {
          last = pct;
          process.stdout.write(`  ${job.id} ${pct * 10}%\n`);
        }
      },
    });
    } catch (err) {
      // Windows can briefly lock the finished file (virus scan / indexer) during Remotion's final rename.
      if (!String(err).includes('EBUSY')) throw err;
      const dir = path.dirname(out);
      const pending = fs.readdirSync(dir).find((f) => f.startsWith(path.basename(out) + '.') && f.endsWith('.remotion-in-progress'));
      if (!pending) throw err;
      let moved = false;
      for (let attempt = 0; attempt < 40 && !moved; attempt++) {
        await new Promise((r) => setTimeout(r, 500));
        try {
          fs.renameSync(path.join(dir, pending), out);
          moved = true;
        } catch {}
      }
      if (!moved) throw err;
      console.log(`  ${job.id}: recovered from a locked-file rename`);
    }
  }
  if (!frames && composition.props?.sidecar) {
    const sidecar = {composition: job.composition, props: {...job.props}, width: composition.width, height: composition.height, durationInFrames: composition.durationInFrames, ...composition.props.sidecar};
    fs.writeFileSync(out.replace(/\.mp4$/, '.json'), JSON.stringify(sidecar, null, 1));
  }
  const seconds = (Date.now() - started) / 1000;
  log.push({id: job.id, out: job.out, frames: composition.durationInFrames, seconds});
  console.log(`done ${job.id}: ${composition.durationInFrames} frames in ${seconds.toFixed(1)} s`);
}
fs.writeFileSync(jobFile.replace(/\.json$/, '.log.json'), JSON.stringify(log, null, 2));
// Each bundle copies public/ into the temp folder (hundreds of MB); remove it so runs don't fill the disk.
try {
  fs.rmSync(serveUrl, {recursive: true, force: true});
} catch (err) {
  console.warn(`could not remove bundle ${serveUrl}: ${err.message}`);
}
