// Varjoluotain browser core: the same light-transport model as the Python
// package, smaller grid (18 x 24 screen patches, 42 x 42 wall pixels).
// Pure functions, no DOM; tested under Node (tests/core.test.mjs).
const VJ = (() => {
  const ROWS = 18, COLS = 24, P = ROWS * COLS, N = 42, M = N * N, SUB = 3;
  const D = 1.03;                                        // screen to wall (m)
  const SX0 = 0.0162, SX1 = 0.4290, SZ0 = 0.1282, SZ1 = 0.4445;   // lit screen area
  const WX0 = 0.521, WZ0 = 0.048, WW = 0.4372;          // camera's view of the wall
  const PLATE = 0.075, STAND = 0.0055;
  const LCD_PZ = 18, LCD_PX = 1;                        // LCD brightness vs viewing angle

  const wallX = new Float64Array(N), wallZrow = new Float64Array(N);
  for (let c = 0; c < N; c++) wallX[c] = WX0 + c * WW / (N - 1);
  for (let r = 0; r < N; r++) wallZrow[r] = WZ0 + WW - r * WW / (N - 1);   // row 0 = top

  function rects(occ) {
    if (!occ || !occ.on) return [];
    const out = [[occ.x, occ.x + PLATE, occ.z, occ.z + PLATE, occ.y]];
    if (occ.stand) {
      const cx = occ.x + PLATE / 2;
      out.push([cx - STAND / 2, cx + STAND / 2, 0, occ.z, occ.y]);
    }
    return out;
  }

  // A[m * P + p]: light reaching wall pixel m from screen patch p at unit brightness
  function buildA(occ) {
    const A = new Float64Array(M * P), R = rects(occ), nR = R.length, D2 = D * D;
    const bw = (SX1 - SX0) / COLS, bh = (SZ1 - SZ0) / ROWS;
    const inx = new Uint8Array(Math.max(1, nR) * N), inz = new Uint8Array(Math.max(1, nR) * N);
    const mx = new Float64Array(N), mz = new Float64Array(N);
    const dx2 = new Float64Array(N), dz2 = new Float64Array(N), col = new Float64Array(M);
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      const p = i * COLS + j, mc = COLS - 1 - j;          // viewer's left = largest x
      const x0 = SX0 + mc * bw, z1 = SZ1 - i * bh;
      const cx = x0 + bw / 2, cz = z1 - bh / 2;
      for (let c = 0; c < N; c++) mx[c] = Math.pow(Math.cos(Math.atan((cx - wallX[c]) / D)), LCD_PX);
      for (let r = 0; r < N; r++) mz[r] = Math.pow(Math.cos(Math.atan((cz - wallZrow[r]) / D)), LCD_PZ);
      col.fill(0);
      for (let a = 0; a < SUB; a++) for (let b = 0; b < SUB; b++) {
        const lx = x0 + (b + 0.5) * bw / SUB, lz = z1 - (a + 0.5) * bh / SUB;
        for (let c = 0; c < N; c++) dx2[c] = (wallX[c] - lx) ** 2;
        for (let r = 0; r < N; r++) dz2[r] = (wallZrow[r] - lz) ** 2;
        for (let q = 0; q < nR; q++) {            // shadow of each opaque rectangle on the wall
          const o = R[q], k = D / o[4];
          const X1 = lx + (o[0] - lx) * k, X2 = lx + (o[1] - lx) * k;
          const Z1 = lz + (o[2] - lz) * k, Z2 = lz + (o[3] - lz) * k;
          const xa = Math.min(X1, X2), xb = Math.max(X1, X2), za = Math.min(Z1, Z2), zb = Math.max(Z1, Z2);
          for (let c = 0; c < N; c++) inx[q * N + c] = wallX[c] >= xa && wallX[c] <= xb ? 1 : 0;
          for (let r = 0; r < N; r++) inz[q * N + r] = wallZrow[r] >= za && wallZrow[r] <= zb ? 1 : 0;
        }
        for (let r = 0; r < N; r++) {
          const zr = dz2[r] + D2, mzr = mz[r], row = r * N;
          for (let c = 0; c < N; c++) {
            let blocked = 0;
            for (let q = 0; q < nR; q++) if (inz[q * N + r] && inx[q * N + c]) { blocked = 1; break; }
            if (blocked) continue;
            const r2 = dx2[c] + zr;
            col[row + c] += D2 / (r2 * r2) * mx[c] * mzr;
          }
        }
      }
      for (let m = 0; m < M; m++) A[m * P + p] = col[m];
    }
    return A;
  }

  // orthonormal room-light basis: constant, left-right ramp, up-down ramp
  const Q = (() => {
    const B = [new Float64Array(M), new Float64Array(M), new Float64Array(M)];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const m = r * N + c;
      B[0][m] = 1; B[1][m] = -1 + 2 * c / (N - 1); B[2][m] = 1 - 2 * r / (N - 1);
    }
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < a; b++) {
        let d = 0; for (let m = 0; m < M; m++) d += B[a][m] * B[b][m];
        for (let m = 0; m < M; m++) B[a][m] -= d * B[b][m];
      }
      let n = 0; for (let m = 0; m < M; m++) n += B[a][m] ** 2;
      n = Math.sqrt(n); for (let m = 0; m < M; m++) B[a][m] /= n;
    }
    return B;
  })();

  function removeRoomLight(v) {                // v: length-M vector, returns a copy
    const out = Float64Array.from(v);
    for (const q of Q) {
      let d = 0; for (let m = 0; m < M; m++) d += q[m] * out[m];
      for (let m = 0; m < M; m++) out[m] -= d * q[m];
    }
    return out;
  }

  function removeRoomLightA(A) {
    const At = Float64Array.from(A);
    for (const q of Q) {
      const d = new Float64Array(P);
      for (let m = 0; m < M; m++) { const qm = q[m]; if (!qm) continue; for (let p = 0; p < P; p++) d[p] += qm * At[m * P + p]; }
      for (let m = 0; m < M; m++) { const qm = q[m]; for (let p = 0; p < P; p++) At[m * P + p] -= qm * d[p]; }
    }
    return At;
  }

  function gram(At) {                           // G = At' At  (P x P, symmetric)
    const G = new Float64Array(P * P);
    for (let m = 0; m < M; m++) {
      const off = m * P;
      for (let a = 0; a < P; a++) {
        const ra = At[off + a];
        if (ra === 0) continue;
        const ga = a * P;
        for (let b = a; b < P; b++) G[ga + b] += ra * At[off + b];
      }
    }
    for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) G[a * P + b] = G[b * P + a];
    return G;
  }

  function whitePeak(A) {
    let best = 0;
    for (let m = 0; m < M; m++) { let s = 0; for (let p = 0; p < P; p++) s += A[m * P + p]; if (s > best) best = s; }
    return best;
  }

  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng(); while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // photo of the wall in electrons: k * A x + room light + shot noise
  function photograph(A, scene, k, photons, seed = 1) {
    const rng = mulberry32(seed), photo = [], sigma = [];
    for (let ch = 0; ch < 3; ch++) {
      const y = new Float64Array(M); let mean = 0;
      for (let m = 0; m < M; m++) {
        let s = 0; const off = m * P;
        for (let p = 0; p < P; p++) s += A[off + p] * scene[p * 3 + ch];
        const r = Math.floor(m / N), c = m % N;
        const amb = 0.2 * photons * (1 + 0.3 * (0.8 * (-1 + 2 * c / (N - 1)) + 0.6 * (1 - 2 * r / (N - 1))));
        const mu = k * s + amb;
        y[m] = mu + Math.sqrt(Math.max(mu, 0)) * randn(rng);
        mean += mu;
      }
      photo.push(y); sigma.push(Math.sqrt(mean / M));
    }
    return { photo, sigma };
  }

  // ---------- total-variation reconstruction (FISTA + Beck-Teboulle prox) ----------
  function Lop(p, q, out) {
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      const t = i * COLS + j;
      let v = p[t] + q[t];
      if (i > 0) v -= p[t - COLS];
      if (j > 0) v -= q[t - 1];
      out[t] = v;
    }
  }
  function LtOp(x, p, q) {
    for (let i = 0; i < ROWS; i++) for (let j = 0; j < COLS; j++) {
      const t = i * COLS + j;
      p[t] = i < ROWS - 1 ? x[t] - x[t + COLS] : 0;
      q[t] = j < COLS - 1 ? x[t] - x[t + 1] : 0;
    }
  }
  function proxTV(bvec, lam, iters, warm) {
    const r = warm ? warm.r : new Float64Array(P), s = warm ? warm.s : new Float64Array(P);
    let pOld = Float64Array.from(r), qOld = Float64Array.from(s), tOld = 1;
    const sol = new Float64Array(P), Lrs = new Float64Array(P), tr = new Float64Array(P), ts = new Float64Array(P);
    const p = new Float64Array(P), q = new Float64Array(P);
    for (let it = 0; it < iters; it++) {
      Lop(r, s, Lrs);
      for (let t = 0; t < P; t++) sol[t] = Math.max(bvec[t] - lam * Lrs[t], 0);
      LtOp(sol, tr, ts);
      for (let t = 0; t < P; t++) {
        const rr = r[t] + tr[t] / (8 * lam), ss = s[t] + ts[t] / (8 * lam);
        const w = Math.max(1, Math.hypot(rr, ss));
        p[t] = rr / w; q[t] = ss / w;
      }
      const tNew = (1 + Math.sqrt(1 + 4 * tOld * tOld)) / 2, f = (tOld - 1) / tNew;
      for (let t = 0; t < P; t++) { r[t] = p[t] + f * (p[t] - pOld[t]); s[t] = q[t] + f * (q[t] - qOld[t]); }
      pOld = Float64Array.from(p); qOld = Float64Array.from(q); tOld = tNew;
    }
    Lop(pOld, qOld, Lrs);
    for (let t = 0; t < P; t++) sol[t] = Math.max(bvec[t] - lam * Lrs[t], 0);
    return { x: sol, warm: { r: pOld, s: qOld } };
  }

  function lmax(G, iters = 60) {
    let v = new Float64Array(P).fill(1 / Math.sqrt(P)), lam = 0;
    const w = new Float64Array(P);
    for (let it = 0; it < iters; it++) {
      for (let a = 0; a < P; a++) { let s = 0; const off = a * P; for (let b = 0; b < P; b++) s += G[off + b] * v[b]; w[a] = s; }
      let n = 0; for (let a = 0; a < P; a++) n += w[a] * w[a];
      n = Math.sqrt(n); lam = n;
      for (let a = 0; a < P; a++) v[a] = w[a] / n;
    }
    return lam;
  }

  // one channel: minimise 0.5 x'(s^2 G)x - (s At'y)'x + lam TV(x), x >= 0
  function reconstructChannel(G, Lg, At, y, scale, lamRel, iters = 140, x0 = null) {
    const ys = removeRoomLight(y);
    const b = new Float64Array(P);
    for (let m = 0; m < M; m++) { const ym = ys[m]; if (!ym) continue; const off = m * P; for (let p = 0; p < P; p++) b[p] += At[off + p] * ym; }
    let bmax = 0; for (let p = 0; p < P; p++) { b[p] *= scale; bmax = Math.max(bmax, Math.abs(b[p])); }
    const L = scale * scale * Lg, lam = lamRel * bmax;
    let x = x0 ? Float64Array.from(x0) : new Float64Array(P), yv = Float64Array.from(x), t = 1, warm = null;
    const g = new Float64Array(P), z = new Float64Array(P);
    for (let it = 0; it < iters; it++) {
      for (let a = 0; a < P; a++) { let s = 0; const off = a * P; for (let c = 0; c < P; c++) s += G[off + c] * yv[c]; g[a] = scale * scale * s - b[a]; }
      for (let a = 0; a < P; a++) z[a] = yv[a] - g[a] / L;
      const pr = proxTV(z, lam / L, 12, warm); warm = pr.warm;
      const tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2, f = (t - 1) / tn;
      for (let a = 0; a < P; a++) { yv[a] = pr.x[a] + f * (pr.x[a] - x[a]); }
      x = pr.x; t = tn;
    }
    return x;
  }

  // ---------- eigenvalues of a symmetric matrix (Householder + implicit QL) ----------
  function eigvalsSym(Gflat, n) {
    const a = []; for (let i = 0; i < n; i++) a.push(Float64Array.from(Gflat.subarray(i * n, (i + 1) * n)));
    const d = new Float64Array(n), e = new Float64Array(n);
    for (let i = n - 1; i > 0; i--) {
      const l = i - 1; let h = 0, scale = 0;
      if (l > 0) {
        for (let k = 0; k <= l; k++) scale += Math.abs(a[i][k]);
        if (scale === 0) e[i] = a[i][l];
        else {
          for (let k = 0; k <= l; k++) { a[i][k] /= scale; h += a[i][k] * a[i][k]; }
          let f = a[i][l];
          let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
          e[i] = scale * g; h -= f * g; a[i][l] = f - g; f = 0;
          for (let j = 0; j <= l; j++) {
            g = 0;
            for (let k = 0; k <= j; k++) g += a[j][k] * a[i][k];
            for (let k = j + 1; k <= l; k++) g += a[k][j] * a[i][k];
            e[j] = g / h; f += e[j] * a[i][j];
          }
          const hh = f / (h + h);
          for (let j = 0; j <= l; j++) {
            f = a[i][j]; e[j] = g = e[j] - hh * f;
            for (let k = 0; k <= j; k++) a[j][k] -= f * e[k] + g * a[i][k];
          }
        }
      } else e[i] = a[i][l];
      d[i] = h;
    }
    for (let i = 0; i < n; i++) d[i] = a[i][i];
    for (let i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    for (let l = 0; l < n; l++) {
      let iter = 0, m;
      do {
        for (m = l; m < n - 1; m++) {
          const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
          if (Math.abs(e[m]) <= 1e-15 * dd) break;
        }
        if (m !== l) {
          if (iter++ === 60) break;
          let g = (d[l + 1] - d[l]) / (2 * e[l]);
          let r = Math.hypot(g, 1);
          g = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));
          let s = 1, c = 1, p = 0, i, underflow = false;
          for (i = m - 1; i >= l; i--) {
            let f = s * e[i]; const b = c * e[i];
            e[i + 1] = r = Math.hypot(f, g);
            if (r === 0) { d[i + 1] -= p; e[m] = 0; underflow = true; break; }
            s = f / r; c = g / r; g = d[i + 1] - p; r = (d[i] - g) * s + 2 * c * b;
            d[i + 1] = g + (p = s * r); g = c * r - b;
          }
          if (underflow) continue;
          d[l] -= p; e[l] = g; e[m] = 0;
        }
      } while (m !== l);
    }
    return Array.from(d).sort((x, y) => y - x);
  }

  return { ROWS, COLS, P, N, M, D, SX0, SX1, SZ0, SZ1, WX0, WZ0, WW, PLATE, STAND,
           buildA, removeRoomLight, removeRoomLightA, gram, whitePeak, photograph,
           reconstructChannel, lmax, eigvalsSym, rects };
})();
if (typeof module !== "undefined") module.exports = VJ;
