import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("production build contains the local and hosted StitchNet shell", async () => {
  const html = await readFile(new URL("dist/index.html", root), "utf8");
  const source = await readFile(new URL("src/App.tsx", root), "utf8");
  const assets = await readdir(new URL("dist/assets/", root));

  assert.match(html, /<title>StitchNet Laboratory · Microscopy Image Stitching<\/title>/i);
  assert.match(html, /confidence-aware microscopy image stitching/i);
  assert.match(html, /application\/ld\+json/i);
  assert.match(html, /SoftwareApplication/i);
  assert.ok(assets.some((name) => name.endsWith(".js")));
  assert.ok(assets.some((name) => name.endsWith(".css")));
  assert.match(source, /Turn overlapping fields into one trustworthy view/i);
  assert.match(source, /Research use only/i);
  assert.match(source, /Images stay on this machine/i);
  assert.match(source, /Read-only hosted preview/i);
  assert.match(source, /Verified H&E sample/i);
  assert.match(source, /Confidence is evidence, not a diagnosis/i);
  assert.doesNotMatch(source, /react-loading-skeleton|Starter Project/i);
  await access(new URL("dist/favicon.svg", root));
  await access(new URL("dist/social-preview.png", root));
  await access(new URL("dist/demo/preview.jpg", root));
  await access(new URL("dist/demo/quality-report.json", root));
});
