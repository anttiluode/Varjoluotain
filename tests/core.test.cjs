// Node checks for the browser core: timing, eigenvalues vs a dense reference,
// and that a scene is recovered with the occluder and not without it.
const VJ = require('../site/core.js');
const fs = require('fs');
const t0 = Date.now();
const occ = { x: 0.475, y: 0.570, z: 0.214, on: true, stand: true };
const A = VJ.buildA(occ);
const tBuild = Date.now() - t0;
const At = VJ.removeRoomLightA(A);
let t1 = Date.now(); const G = VJ.gram(At); const tGram = Date.now() - t1;
t1 = Date.now(); const ev = VJ.eigvalsSym(G, VJ.P); const tEig = Date.now() - t1;
// a smooth test scene with an edge
const scene = new Float64Array(VJ.P * 3);
for (let i = 0; i < VJ.ROWS; i++) for (let j = 0; j < VJ.COLS; j++) {
  const p = i * VJ.COLS + j;
  scene[p * 3] = j < 12 ? 0.9 : 0.1; scene[p * 3 + 1] = i / VJ.ROWS; scene[p * 3 + 2] = (i + j) % 6 < 3 ? 0.8 : 0.2;
}
const Aempty = VJ.buildA({ on: false });
const photons = 1e6, k = photons / VJ.whitePeak(Aempty);
function psnrOf(Amat, Gm, Atm) {
  const { photo } = VJ.photograph(Amat, scene, k, photons, 3);
  const L = VJ.lmax(Gm);
  let mse = 0;
  for (let ch = 0; ch < 3; ch++) {
    const x = VJ.reconstructChannel(Gm, L, Atm, photo[ch], k, 1e-4, 150);
    for (let p = 0; p < VJ.P; p++) mse += (Math.min(1, x[p]) - scene[p * 3 + ch]) ** 2;
  }
  return 10 * Math.log10(1 / (mse / (3 * VJ.P)));
}
t1 = Date.now(); const ps = psnrOf(A, G, At); const tRec = Date.now() - t1;
const At0 = VJ.removeRoomLightA(Aempty), G0 = VJ.gram(At0);
const ps0 = psnrOf(Aempty, G0, At0);
fs.writeFileSync('/tmp/G_js.bin', Buffer.from(G.buffer));
fs.writeFileSync('/tmp/ev_js.json', JSON.stringify(ev.slice(0, 20).concat(ev.slice(-5))));
console.log(JSON.stringify({ buildA_ms: tBuild, gram_ms: tGram, eig_ms: tEig, recon3_ms: tRec,
  psnr_with_occluder: +ps.toFixed(2), psnr_without: +ps0.toFixed(2), top_eig: ev[0], min_eig: ev[ev.length - 1] }));
if (!(ps > ps0 + 5)) { console.error('FAIL: occluder should help'); process.exit(1); }
