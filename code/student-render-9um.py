"""day1-render-9um.py - c0841-9um.py's native 9.366 um render, made reusable for any PHerc0841 mesh.

WHY. The day-1 plan (geom/day1-9um/design.json, key "choice", step 3) runs a reader frozen on w00big
once on auto_grown_20260220174252405, text no model has seen. That input has to be made exactly the way
w00big's was (c0841-9um.py), because w00big is where the recipe is chosen: a 2.403 um tifxyz mesh is
carried into the 9.366 um frame of volume 20250821151531 by reg-pherc0841-affine.json, sampled
trilinear at 9.61 um in plane (4 canvas px of the mesh's 20-px-per-cell 2.403 um canvas) and at 41
layers 9.6 um apart along the mesh normal, and stored in the normal's order, so that a reader's 'rev'
means what it meant on w00big.

WHAT IS NEW, each one checked rather than assumed:
  - Mesh, crop (in mesh cells, end-exclusive) and output are arguments. The render goes tile by tile
    through the shared chunk cache, so a large frame never needs one dense box. --compare-stack
    re-renders a frame that c0841-9um.py already rendered and must reproduce it.
  - Cells holding the -1 sentinel are masked; a pixel is kept only when its four bilinear corners,
    c0841's three rounded cells and the 3x3 cells behind its normal are all valid. On a fully valid
    mesh (w00big) that is exactly c0841's mask.
  - THE STACK ORDER IS SET BY GEOMETRY, never by a map or a label. The normal's sense relative to the
    scroll is measured for this mesh and for the reference frame (w00big), two independent ways: the
    radial direction from the scroll's centroid axis in the level-4 volume, and the normals of the
    nearest reference sheet (adjacent windings are near parallel). The stack is flipped when the
    sense differs from the reference's. If the two ways disagree, or either is ambiguous, the run
    stops before writing a stack.
  - 2.403 um canvases (ScrollPrize's ink map, labels, masks; shape = 20 x the mesh grid) come onto
    the grid BY INDEX: render px (i, j) is the mean of canvas block [R0+4i, R0+4i+4) x [C0+4j,
    C0+4j+4), R0 = 20 x crop row, C0 = 20 x crop col; saved /255 as float32.
  - Gate 1, model-free (--map NAME): Pearson r of the direct render (layers 18-22 mean, c0841's) with
    that canvas, at zero shift and at c0841's 48 rolled shifts of 1-3 mm (the null), plain and
    smoothed sigma 4 px. It must stay inside the null (w00big: +0.032 against null max +0.066).
  - Cross-check, model-free (--check-mesh): distance from our affine placement to ScrollPrize's own
    9.366 um mesh of the same segment, split into the normal and in-plane parts; and the catalog
    matrix's placement against the same mesh, which says whether the 2.403 um mesh came from it.

WHAT WOULD MAKE IT WORTHLESS, decided before running: a --compare-stack mismatch on w00big (the
render is not c0841's); a direct render that correlates with the 2.4 um map beyond its null (density,
not ink, would then carry the signal); an order the two geometric tests disagree on.

Usage:
  day1-render-9um.py --mesh DIR --crop r0 r1 c0 c1 --out DIR --name NAME
                     [--canvas name=path ...] [--map name] [--check-mesh DIR] [--catalog-mesh-from 9366|none]
                     [--orient-ref DIR --orient-ref-crop r0 r1 c0 c1] [--no-orient]
                     [--compare-stack npy --compare-mask npy] [--dry]
  a canvas path is a .tif or a zarr v3 array directory (level 0).
CPU and network only (no model). Streams public data (CC BY-NC 4.0) into the chunk cache; publishes
nothing. Run from <workdir>.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument('--mesh', required=True)
ap.add_argument('--crop', type=int, nargs=4, required=True, metavar=('R0', 'R1', 'C0', 'C1'))
ap.add_argument('--out', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--affine', default='reg-pherc0841-affine.json')
ap.add_argument('--nl', type=int, default=41)
ap.add_argument('--pool', type=int, default=4)
ap.add_argument('--canvas', action='append', default=[])
ap.add_argument('--map', default=None)
ap.add_argument('--check-mesh', default=None)
ap.add_argument('--orient-ref', default=None)
ap.add_argument('--orient-ref-crop', type=int, nargs=4, default=None)
ap.add_argument('--no-orient', action='store_true')
ap.add_argument('--axis-vol', default='geom/l0-1203-render/audit/build-deciding-test/reg-pherc0841/20250821151531-L4.npy')
ap.add_argument('--axis-scale', type=int, default=16)
ap.add_argument('--compare-stack', default=None)
ap.add_argument('--compare-mask', default=None)
ap.add_argument('--dry', action='store_true')
ap.add_argument('--tile', type=int, default=256)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
OUT = os.path.join(a.out, '%s-run.txt' % a.name)
import runfile
runfile.claim(OUT)

B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/'
VOL = 'PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr/0/'
CACHE = os.path.join(os.environ.get('SCROLL_CACHE', 'scroll-data/chunkcache'), VOL.strip('/').replace('/', '_'))
UM = 9.366
STEP_N = 9.6 / UM
NL, POOL, CH = a.nl, a.pool, 128
t0 = time.time()
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def fetch(url, timeout=300):
    req = urllib.request.Request(url, headers={'User-Agent': 'scrolls-fl'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ensure(key):
    """Put one 128^3 chunk of the 9.366 um volume in the shared cache. 'cached', 'fetched' or 'absent'."""
    cp = os.path.join(CACHE, '%d_%d_%d' % key)
    if os.path.exists(cp):
        return 'cached'
    for attempt in range(4):
        try:
            raw = fetch(B + VOL + '/'.join(map(str, key)))
            break
        except Exception as e:
            if '404' in str(e) or '403' in str(e):
                os.makedirs(CACHE, exist_ok=True)
                open(cp, 'wb').close()
                return 'absent'
            if attempt == 3:
                raise
            time.sleep(2 + 4 * attempt)
    os.makedirs(CACHE, exist_ok=True)
    with open(cp + '.part', 'wb') as fh:
        fh.write(raw)
    os.replace(cp + '.part', cp)
    return 'fetched'


def load_chunk(key):
    cp = os.path.join(CACHE, '%d_%d_%d' % key)
    raw = open(cp, 'rb').read()
    return np.frombuffer(raw, np.uint8).reshape(CH, CH, CH) if raw else None


def bilinear(A, ii, jj):
    i1 = np.clip(np.floor(ii).astype(int), 0, A.shape[0] - 2)
    j1 = np.clip(np.floor(jj).astype(int), 0, A.shape[1] - 2)
    di, dj = ii - i1, jj - j1
    return (A[i1, j1] * (1 - di) * (1 - dj) + A[i1 + 1, j1] * di * (1 - dj)
            + A[i1, j1 + 1] * (1 - di) * dj + A[i1 + 1, j1 + 1] * di * dj)


def stretch(A, mask):
    v = A[mask]
    if v.size == 0:
        return np.zeros(A.shape, np.uint8)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    out = np.clip((A - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    out[~mask] = 0
    return out.astype(np.uint8)


def sha(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]


aff = json.load(open(a.affine))
AFF = np.array(aff['fine_to_coarse_affine'])


def load_mesh(d, crop=None):
    X, Y, Z = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    if crop is not None:
        r0, r1, c0, c1 = crop
        X, Y, Z = X[r0:r1, c0:c1], Y[r0:r1, c0:c1], Z[r0:r1, c0:c1]
    valid = (X > 0) & (Y > 0) & (Z > 0)
    return X, Y, Z, valid


def to_coarse(X, Y, Z):
    P = np.stack([Z, Y, X], -1).reshape(-1, 3)
    Q = np.append(P, np.ones((len(P), 1)), 1) @ AFF
    return np.stack([Q[:, k].reshape(X.shape) for k in range(3)], -1)        # (h, w, 3) z, y, x


def normals(Pm, valid):
    """c0841's convention: cross(d/drow, d/dcol) of the coarse (z, y, x) mesh; zero where the 3x3 is not valid."""
    N = np.cross(np.gradient(Pm, axis=0), np.gradient(Pm, axis=1))
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    v3 = binary_erosion(valid, np.ones((3, 3), bool), border_value=1)
    N[~v3] = 0
    return N, v3


# ---- the mesh, carried into the 9.366 um frame --------------------------------------------------
say('day1-render-9um.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
say('  argv: %s' % ' '.join(sys.argv[1:]))
say('  registration %s (sha1 %s): median %.1f um in-sample, %.1f um held out, %d blocks'
    % (a.affine, sha(a.affine), aff['median_residual_um'], aff['median_loo_um'], aff['blocks']))
X, Y, Z, valid = load_mesh(a.mesh, a.crop)
FULL = tifffile.imread(os.path.join(a.mesh, 'x.tif')).shape
h, w = X.shape
say('  mesh %s crop rows %d-%d cols %d-%d: %dx%d cells, %d valid (%.1f%%)'
    % (a.mesh, a.crop[0], a.crop[1], a.crop[2], a.crop[3], h, w, valid.sum(), 100 * valid.mean()))
Pm = to_coarse(X, Y, Z)
Pm[~valid] = 0
N, v3 = normals(Pm, valid)
H, W = (h - 1) * 20 // POOL, (w - 1) * 20 // POOL
ii = (np.arange(H) * POOL + (POOL - 1) / 2) / 20.0
jj = (np.arange(W) * POOL + (POOL - 1) / 2) / 20.0
JJ, II = np.meshgrid(jj, ii)
ci = np.clip(np.rint(II).astype(int), 0, h - 1)
cj = np.clip(np.rint(JJ).astype(int), 0, w - 1)
mask = valid[ci, cj] & valid[np.clip(ci + 1, 0, h - 1), cj] & valid[ci, np.clip(cj + 1, 0, w - 1)]   # c0841's rule
fi = np.clip(np.floor(II).astype(int), 0, h - 2)
fj = np.clip(np.floor(JJ).astype(int), 0, w - 2)
for di in (0, 1):
    for dj in (0, 1):
        mask &= valid[fi + di, fj + dj] & v3[fi + di, fj + dj]
say('  grid %dx%d at %.2f um in plane, %d layers at %.2f um (%.3f voxels); %d px in the mask (%.2f cm2)'
    % (H, W, 2.403 * POOL, NL, 9.6, STEP_N, mask.sum(), mask.sum() * (2.403 * POOL * 1e-4) ** 2))
pz, py, px = (bilinear(Pm[..., c], II, JJ) for c in range(3))
n = np.stack([bilinear(N[..., c], II, JJ) for c in range(3)], -1)
n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)

# ---- the order, from geometry only --------------------------------------------------------------
flip = False
orient = {}
if not a.no_orient and a.orient_ref:
    L4 = np.load(a.axis_vol, mmap_mode='r')
    S = a.axis_scale
    cen = np.full((L4.shape[0], 2), np.nan)
    for zz in range(L4.shape[0]):
        sl = np.asarray(L4[zz])
        yy, xx = np.nonzero(sl > 0)
        if len(yy) > 100:
            cen[zz] = (yy.mean() * S + (S - 1) / 2, xx.mean() * S + (S - 1) / 2)

    def radial_sense(P, Nn, ok):
        zi = np.clip((P[..., 0][ok] // S).astype(int), 0, L4.shape[0] - 1)
        c = cen[zi]
        r = np.stack([P[..., 1][ok] - c[:, 0], P[..., 2][ok] - c[:, 1]], 1)
        nyx = Nn[..., 1:][ok]
        good = np.isfinite(r).all(1) & (np.linalg.norm(nyx, axis=1) > 0.3)
        s = (r[good] * nyx[good]).sum(1) / np.linalg.norm(r[good], axis=1) / np.linalg.norm(nyx[good], axis=1)
        rad = np.linalg.norm(r[good], axis=1) * UM / 1000
        return s, rad
    Xr, Yr, Zr, vr = load_mesh(a.orient_ref)
    Pr = to_coarse(Xr, Yr, Zr)
    Pr[~vr] = 0
    Nr, v3r = normals(Pr, vr)
    rc = a.orient_ref_crop or (0, Xr.shape[0], 0, Xr.shape[1])
    s_t, rad_t = radial_sense(Pm, N, v3)
    s_r, rad_r = radial_sense(Pr[rc[0]:rc[1], rc[2]:rc[3]], Nr[rc[0]:rc[1], rc[2]:rc[3]], v3r[rc[0]:rc[1], rc[2]:rc[3]])
    say('  ORDER, way 1 - normal . outward radial (scroll axis = centroid of the masked level-4 volume per slice):')
    say('    this mesh:      median %+.3f, frac > 0 %.3f, n %d, radius median %.1f mm' % (np.median(s_t), (s_t > 0).mean(), len(s_t), np.median(rad_t)))
    say('    reference crop: median %+.3f, frac > 0 %.3f, n %d, radius median %.1f mm' % (np.median(s_r), (s_r > 0).mean(), len(s_r), np.median(rad_r)))
    sense_t = np.sign(np.median(s_t))
    sense_r = np.sign(np.median(s_r))
    # ambiguous: a weak median, or more than a fifth of the cells pointing the other way
    amb1 = bool(min(abs(np.median(s_t)), abs(np.median(s_r))) < 0.3
                or min((s_t > 0).mean(), (s_t <= 0).mean()) > 0.2 or min((s_r > 0).mean(), (s_r <= 0).mean()) > 0.2)
    flip1 = bool(sense_t != sense_r)
    # way 2: the nearest reference sheet's normal (adjacent windings are near parallel)
    okr = v3r
    Tr = cKDTree(Pr[okr])
    Nrv = Nr[okr]
    sel = np.argwhere(v3)[::7]
    d, k = Tr.query(Pm[sel[:, 0], sel[:, 1]])
    dots = (N[sel[:, 0], sel[:, 1]] * Nrv[k]).sum(1)
    near = d < 100                        # coarse voxels, 0.94 mm
    say('  ORDER, way 2 - normal . normal of the nearest reference-mesh point (whole reference mesh, same grid sense as its crop):')
    say('    distance to the reference sheet: median %.0f, p10 %.0f coarse voxels; %d of %d sampled cells within 100'
        % (np.median(d), np.percentile(d, 10), near.sum(), len(d)))
    # rule fixed before the first run: the cells within 100 voxels if there are at least 200 of them,
    # otherwise the nearest 5% of the sampled cells
    if near.sum() >= 200:
        dn = dots[near]
        say('    within 100 voxels: dot median %+.3f, frac > 0 %.3f' % (np.median(dn), (dn > 0).mean()))
    else:
        cut = np.percentile(d, 5)
        dn = dots[d <= cut]
        say('    fewer than 200 within 100 voxels; the nearest 5%% (d <= %.0f voxels, %d cells): dot median %+.3f, frac > 0 %.3f'
            % (cut, len(dn), np.median(dn), (dn > 0).mean()))
    flip2 = bool(np.median(dn) < 0)
    amb2 = bool(abs(np.median(dn)) < 0.5 or 0.1 < (dn > 0).mean() < 0.9)
    say('    way 1 says flip=%s%s; way 2 says flip=%s%s' % (flip1, ' (AMBIGUOUS)' if amb1 else '', flip2, ' (AMBIGUOUS)' if amb2 else ''))
    orient = {'way1_target_median': float(np.median(s_t)), 'way1_target_frac_pos': float((s_t > 0).mean()),
              'way1_ref_median': float(np.median(s_r)), 'way1_ref_frac_pos': float((s_r > 0).mean()),
              'way2_dot_median': float(np.median(dn)), 'way2_frac_pos': float((dn > 0).mean()), 'way2_n_near': int(near.sum()),
              'flip_way1': flip1, 'flip_way2': flip2, 'ambiguous_way1': bool(amb1), 'ambiguous_way2': bool(amb2)}
    if flip1 != flip2 or amb1 or amb2:
        say('  STOP: the two geometric ways do not give one clear order; no stack written.')
        runfile.done(OUT, '\n'.join(lines))
        sys.exit(2)
    flip = flip1
    say('  ORDER: %s - the stored stack runs along %s the mesh normal, so it has the reference frame\'s sense and "rev" means what it meant there'
        % ('FLIPPED' if flip else 'kept', 'MINUS' if flip else 'PLUS'))
elif a.no_orient:
    say('  ORDER: not checked (--no-orient); stored along PLUS the mesh normal, c0841\'s convention')
nsign = -1.0 if flip else 1.0

# ---- cross-check against the published 9.366 um mesh of the same segment ----------------------
if a.check_mesh:
    X9, Y9, Z9, v9 = load_mesh(a.check_mesh)
    sub = 4
    r9 = np.arange(0, X9.shape[0] - 1 + 1e-9, 1.0 / sub)
    c9 = np.arange(0, X9.shape[1] - 1 + 1e-9, 1.0 / sub)
    C9, R9 = np.meshgrid(c9, r9)
    f0, g0 = np.clip(np.floor(R9).astype(int), 0, X9.shape[0] - 2), np.clip(np.floor(C9).astype(int), 0, X9.shape[1] - 2)
    ok9 = v9[f0, g0] & v9[f0 + 1, g0] & v9[f0, g0 + 1] & v9[f0 + 1, g0 + 1]
    P9 = np.stack([bilinear(Z9, R9, C9)[ok9], bilinear(Y9, R9, C9)[ok9], bilinear(X9, R9, C9)[ok9]], 1)
    T9 = cKDTree(P9)
    P9g = np.stack([Z9, Y9, X9], -1)
    P9g[~v9] = 0
    N9, _ = normals(P9g, v9)
    N9s = np.stack([bilinear(N9[..., c], R9, C9)[ok9] for c in range(3)], 1)
    N9s /= np.maximum(np.linalg.norm(N9s, axis=1, keepdims=True), 1e-9)
    sel = np.zeros_like(mask)
    sel[::3, ::3] = True
    m = mask & sel
    q = np.stack([pz[m], py[m], px[m]], 1)
    d, k = T9.query(q)
    dn_ = np.abs(((P9[k] - q) * N9s[k]).sum(1))
    say("  CROSS-CHECK vs ScrollPrize's own 9.366 um mesh (%s, sampled at 1/%d cell), %d render px; um:" % (a.check_mesh, sub, len(d)))
    say("    our placement: offset along that mesh's normal median %.1f p95 %.1f max %.1f (nearest sample point: median %.1f)"
        % (np.median(dn_) * UM, np.percentile(dn_, 95) * UM, dn_.max() * UM, np.median(d) * UM))
    cat = json.load(open('geom/metadata-new.json', encoding='utf-8'))['samples']['PHerc0841']['sample']['properties']['volume_transforms']
    Mc = np.array([t['matrix'] for e in cat if e['from_volume_id'] == '20260319124803' for t in e['transforms']][0])
    xyz = np.stack([bilinear(X, II, JJ)[m], bilinear(Y, II, JJ)[m], bilinear(Z, II, JJ)[m]], 1) @ Mc[:, :3].T + Mc[:, 3]
    qc = xyz[:, ::-1]
    dc_, kc = T9.query(qc)
    dnc = np.abs(((P9[kc] - qc) * N9s[kc]).sum(1))
    dq = np.linalg.norm(qc - q, axis=1)
    say("    catalog matrix's placement: offset along that mesh's normal median %.1f p95 %.1f max %.1f"
        % (np.median(dnc) * UM, np.percentile(dnc, 95) * UM, dnc.max() * UM))
    say('    our affine vs the catalog matrix, same points: median %.1f p95 %.1f max %.1f  (w00big: 6.0 8.6 9.3)'
        % (np.median(dq) * UM, np.percentile(dq, 95) * UM, dq.max() * UM))

# ---- chunks ----------------------------------------------------------------------------------------
half = NL // 2
mi = np.nonzero(mask)
mz, my, mx = pz[mi], py[mi], px[mi]
mn = n[mi] * nsign
enc = []
for k in range(-half, half + 1):
    zf = np.floor(mz + k * STEP_N * mn[:, 0]).astype(np.int64)
    yf = np.floor(my + k * STEP_N * mn[:, 1]).astype(np.int64)
    xf = np.floor(mx + k * STEP_N * mn[:, 2]).astype(np.int64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                enc.append(np.unique((((zf + dz) // CH) * 1000 + (yf + dy) // CH) * 1000 + (xf + dx) // CH))
enc = np.unique(np.concatenate(enc))
keys = [(int(e // 1000000), int(e // 1000 % 1000), int(e % 1000)) for e in enc]
have = sum(os.path.exists(os.path.join(CACHE, '%d_%d_%d' % kk)) for kk in keys)
say('  %d chunks touched (%d already cached, %d to fetch, about %.2f GB at 2 MB each)' % (len(keys), have, len(keys) - have, (len(keys) - have) * 2.097e-3))
if a.dry:
    say('  --dry: stopping before any fetch')
    runfile.done(OUT, '\n'.join(lines))
    sys.exit(0)
t1 = time.time()
with ThreadPoolExecutor(16) as ex:
    st = list(ex.map(ensure, keys))
say('  chunks: %s in %.0f s' % ({s: st.count(s) for s in sorted(set(st))}, time.time() - t1))
keyset = set(keys)

# ---- render, tile by tile -------------------------------------------------------------------------
t1 = time.time()
stack = np.zeros((NL, H, W), np.uint8)
T = a.tile
for ti in range(0, H, T):
    for tj in range(0, W, T):
        tm = mask[ti:ti + T, tj:tj + T]
        if not tm.any():
            continue
        ix = np.nonzero(tm)
        tz, ty, tx = pz[ti:ti + T, tj:tj + T][ix], py[ti:ti + T, tj:tj + T][ix], px[ti:ti + T, tj:tj + T][ix]
        tn = n[ti:ti + T, tj:tj + T][ix] * nsign
        lo = np.array([np.inf] * 3)
        hi = -lo
        for k in (-half, half):
            for c, (p0, nc) in enumerate(((tz, tn[:, 0]), (ty, tn[:, 1]), (tx, tn[:, 2]))):
                f = np.floor(p0 + k * STEP_N * nc)
                lo[c] = min(lo[c], f.min())
                hi[c] = max(hi[c], f.max() + 1)
        c_lo = (lo // CH).astype(int)
        c_hi = (hi // CH).astype(int) + 1
        dense = np.zeros(tuple((c_hi - c_lo) * CH), np.uint8)
        for kz in range(c_lo[0], c_hi[0]):
            for ky in range(c_lo[1], c_hi[1]):
                for kx in range(c_lo[2], c_hi[2]):
                    if (kz, ky, kx) not in keyset:
                        continue
                    arr = load_chunk((kz, ky, kx))
                    if arr is None:
                        continue
                    o = (np.array((kz, ky, kx)) - c_lo) * CH
                    dense[o[0]:o[0] + CH, o[1]:o[1] + CH, o[2]:o[2] + CH] = arr
        org = c_lo * CH
        for li, k in enumerate(range(-half, half + 1)):
            cz = tz + k * STEP_N * tn[:, 0] - org[0]
            cy = ty + k * STEP_N * tn[:, 1] - org[1]
            cx = tx + k * STEP_N * tn[:, 2] - org[2]
            z0 = np.clip(np.floor(cz).astype(int), 0, dense.shape[0] - 2)
            y0 = np.clip(np.floor(cy).astype(int), 0, dense.shape[1] - 2)
            x0 = np.clip(np.floor(cx).astype(int), 0, dense.shape[2] - 2)
            fz, fy, fx = cz - z0, cy - y0, cx - x0
            acc = np.zeros(len(cz), np.float32)
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        wgt = (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                        acc += wgt * dense[z0 + dz, y0 + dy, x0 + dx]
            sub_ = stack[li, ti:ti + T, tj:tj + T]
            sub_[ix] = np.clip(np.rint(acc), 0, 255).astype(np.uint8)
        del dense
say('  rendered %s in %.0f s' % (stack.shape, time.time() - t1))
np.save(os.path.join(a.out, '%s-stack.npy' % a.name), stack)
np.save(os.path.join(a.out, '%s-mask.npy' % a.name), mask)
prof = [float(stack[k][mask].mean()) for k in range(NL)]
say('  layer means (9.6 um steps, stored order, centre = mesh) %s' % ' '.join('%.0f' % v for v in prof))
direct = stack[NL // 2 - 2:NL // 2 + 3].astype(np.float32).mean(axis=0)
Image.fromarray(stretch(direct, mask)).save(os.path.join(a.out, '%s-render.png' % a.name))

if a.compare_stack:
    ref = np.load(a.compare_stack, mmap_mode='r')
    refm = np.load(a.compare_mask) if a.compare_mask else None
    if ref.shape != stack.shape:
        say('  COMPARE: shape %s vs %s - MISMATCH' % (ref.shape, stack.shape))
    else:
        dif = np.abs(np.asarray(ref, np.int16) - stack.astype(np.int16))
        say('  COMPARE with %s: voxels differing %d of %d, max |diff| %d; mask equal %s'
            % (a.compare_stack, int((dif > 0).sum()), dif.size, int(dif.max()), None if refm is None else bool((refm == mask).all())))

# ---- 2.403 um canvases by grid index --------------------------------------------------------------
R0, C0 = a.crop[0] * 20, a.crop[2] * 20
grids = {}
for spec in a.canvas:
    name, path = spec.split('=', 1)
    if os.path.isdir(path):
        import zarr
        cv = zarr.open_array(path, mode='r')[:]
    else:
        cv = tifffile.imread(path)
    cv = cv.reshape(cv.shape[-2:]) if cv.ndim > 2 else cv
    blk = cv[R0:R0 + H * POOL, C0:C0 + W * POOL]
    if blk.shape != (H * POOL, W * POOL):
        say('  canvas %s %s: crop runs off the canvas (%s) - skipped' % (name, cv.shape, blk.shape))
        continue
    g = blk.astype(np.float32).reshape(H, POOL, W, POOL).mean((1, 3)) / 255.0
    grids[name] = g
    np.save(os.path.join(a.out, '%s-%s-grid.npy' % (a.name, name)), g)
    say('  canvas %-12s %s (canvas %s, %.2f x the mesh grid): mean over the mask %.4f, frac > 0.5 %.4f'
        % (name, path, cv.shape, cv.shape[0] / FULL[0], g[mask].mean(), (g[mask] > 0.5).mean()))
    del cv, blk

json.dump({'mesh': a.mesh, 'crop_cells': a.crop, 'canvas_origin_px': [R0, C0], 'pool': POOL, 'H': H, 'W': W, 'NL': NL,
           'pitch_um_in_plane': 2.403 * POOL, 'pitch_um_depth': 9.6, 'depth_step_voxels': STEP_N, 'volume': VOL,
           'affine': a.affine, 'affine_sha1': sha(a.affine), 'stored_along': 'minus normal' if flip else 'plus normal',
           'flipped': flip, 'orient': orient, 'centre_layer': NL // 2,
           'render_px_to_canvas': 'render (i, j) = mean of canvas block [R0+%d i, +%d) x [C0+%d j, +%d)' % (POOL, POOL, POOL, POOL),
           'grids': sorted(grids), 'grid_units': 'canvas value / 255, block mean',
           'stack_sha1': hashlib.sha1(stack.tobytes()).hexdigest()[:12]},
          open(os.path.join(a.out, '%s-grid.json' % a.name), 'w'), indent=1)

# ---- gate 1: the direct render against the 2.4 um map, inside its rolled null --------------------
if a.map and a.map in grids:
    ref = grids[a.map] * 255
    E = 40
    M = mask.copy()
    M[:E] = M[-E:] = False
    M[:, :E] = M[:, -E:] = False
    SHIFTS = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]

    def r_at(p, q, dy, dx, sig):
        if sig:
            p = gaussian_filter(p, sig)
            q = gaussian_filter(q, sig)
        ys, ye = max(0, dy), min(H, H + dy)
        xs, xe = max(0, dx), min(W, W + dx)
        pp = p[ys:ye, xs:xe]
        qq = q[ys - dy:ye - dy, xs - dx:xe - dx]
        mm = M[ys:ye, xs:xe] & M[ys - dy:ye - dy, xs - dx:xe - dx]
        if mm.sum() < 1000:
            return np.nan
        return float(np.corrcoef(pp[mm], qq[mm])[0, 1])
    say('')
    say('  GATE 1 (model-free): direct render (layers %d-%d mean) vs %s on the grid, %d px (%.2f cm2) after a %d px edge cut'
        % (NL // 2 - 2, NL // 2 + 2, a.map, M.sum(), M.sum() * (2.403 * POOL * 1e-4) ** 2, E))
    res = {}
    for sig in (0, 4):
        r0 = r_at(direct, ref, 0, 0, sig)
        nl = np.array([r_at(direct, ref, dy, dx, sig) for dy, dx in SHIFTS])
        nl = nl[np.isfinite(nl)]
        z = (r0 - nl.mean()) / max(nl.std(), 1e-9)
        res[sig] = (r0, nl.mean(), nl.max(), z, len(nl))
        say('    sigma %d: r %+.3f | null (%d shifts of 1-3 mm) mean %+.3f max %+.3f | z %.1f -> %s'
            % (sig, r0, len(nl), nl.mean(), nl.max(), z, 'INSIDE the null' if r0 <= nl.max() else 'OUTSIDE the null'))
    say('    reference, w00big (c0841-9um-run.txt, vs the 2.4 um Hecate map): r +0.032, null max +0.066; vs ScrollPrize\'s map r -0.026')
    say('    GATE 1 %s' % ('PASS' if res[0][0] <= res[0][2] else 'FAIL'))
    json.dump({'gate1': {str(s): dict(zip(('r', 'null_mean', 'null_max', 'z', 'n_null'), v)) for s, v in res.items()},
               'pass': bool(res[0][0] <= res[0][2])}, open(os.path.join(a.out, '%s-gate1.json' % a.name), 'w'), indent=1)

say('  %.0f s' % (time.time() - t0))
runfile.done(OUT, '\n'.join(lines))
