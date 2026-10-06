"""fl-register-0841.py - PHerc0841's 2.403 um scan into its NATIVE 9.366 um frame: the control's half
of the deciding test.

WHY. Villa #1739 ruled First Letters evidence must come wholly from the eligible ~9.4 um volume. So
the question every 2.4 um scout find now hangs on is whether ANY 9 um model reads known text from a
native 9 um scan. PHerc0841 is the only control on this desk with both scans and known text (the w00
segment, ScrollPrize's 2.4 um ink map, and Hecate 2.4 um letters in c0841-control-2p4um.png). Our
earlier 9.6 um route (c0841-run.txt) pooled the 2.403 scan's own level 2, never the 9.366 volume, so
it never asked this question. A mesh that lives in the 2.403 frame needs this registration to be
rendered from the 9.366 volume.

HOW. fl-register.py's coarse search (every 2 deg, both hands, FFT NCC, then +-3 deg in quarters)
and fl-register-fine.py's block refinement (blocks at full 9.366 um against the fine scan's level 2
resampled, a least-squares affine, leave-one-out residual), in one script, for this sample. Extra
blocks sit inside the w00 control frame's bounding box, because that is where the map is used.
fl-register.py and fl-register-fine.py are left as they are.

INDEPENDENT CHECK. The October catalog (geom/metadata-new.json) publishes ScrollPrize's own
20260319124803 -> 20250821151531 matrix for this sample. It is not used to fit anything; the fit's
disagreement with it is printed, on the matched blocks and on the w00 control mesh, in um.

WHAT WOULD MAKE IT WORTHLESS, decided before running: a stage-1 best that does not stand clear of
its rivals or of the other hand; a held-out residual near a sheet spacing (~100 um); or a fit that
disagrees with the catalog matrix by more than a sheet on the control mesh.

Usage:  fl-register-0841.py [run.txt] [--fetch-only]
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FETCH_ONLY = '--fetch-only' in sys.argv
if FETCH_ONLY:
    sys.argv.remove('--fetch-only')
OUT = sys.argv[1] if len(sys.argv) > 1 else 'fl-register-0841-run.txt'
import runfile
runfile.claim(OUT)

SAMPLE = 'PHerc0841'
B = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/'
# ids confirmed against geom/metadata-new.json (volumes of PHerc0841) and the data browser page
FINE_ID, COARSE_ID = '20260319124803', '20250821151531'
fine_p = 'PHerc0841/volumes/20260319124803-2.403um-0.2m-77keV-masked.zarr/'
coarse_p = 'PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr/'
FINE_UM, COARSE_UM = 2.403, 9.366
FINE_LV, COARSE_LV = 5, 4
WORK = os.path.join('geom', 'l0-1203-render', 'audit', 'build-deciding-test', 'reg-pherc0841')
AFF_OUT = 'reg-pherc0841-affine.json'
CTRL = 'geom/c0841/mesh0b/w00big'
CH = 128
HALF = 80                       # block half-size in coarse voxels: 160^3, 1.5 mm
SEARCH = 24
dev = os.environ.get('REG_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
t0 = time.time()
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def fetch(url, timeout=300):
    req = urllib.request.Request(url, headers={'User-Agent': 'scrolls-fl'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def level(prefix, lv):
    """A whole pyramid level as uint8, cached as .npy under WORK."""
    cache = os.path.join(WORK, '%s-L%d.npy' % (prefix.rstrip('/').split('/')[-1][:14], lv))
    if os.path.exists(cache):
        return np.load(cache)
    za = json.loads(fetch(B + prefix + '%d/.zarray' % lv))
    shp = za['shape']
    nb = [int(np.ceil(s / CH)) for s in shp]
    out = np.zeros(shp, np.uint8)

    def one(k):
        try:
            raw = np.frombuffer(fetch(B + prefix + '%d/%d/%d/%d' % ((lv,) + k)), np.uint8)
        except Exception:
            return 0
        a = raw.reshape(CH, CH, CH)
        z, y, x = (k[i] * CH for i in range(3))
        sz = out[z:z + CH, y:y + CH, x:x + CH].shape
        out[z:z + CH, y:y + CH, x:x + CH] = a[:sz[0], :sz[1], :sz[2]]
        return 1
    keys = [(a, b, c) for a in range(nb[0]) for b in range(nb[1]) for c in range(nb[2])]
    with ThreadPoolExecutor(16) as ex:
        got = sum(ex.map(one, keys))
    say('  %s L%d %s: %d of %d chunks' % (prefix.split('/')[-2][:14], lv, shp, got, len(keys)))
    np.save(cache, out)
    return out


def box(prefix, lv, lo, hi):
    """Voxels [lo, hi) of a pyramid level, uint8, zeros where the bucket has no chunk."""
    lo, hi = np.array(lo), np.array(hi)
    out = np.zeros(tuple(hi - lo), np.uint8)
    c0, c1 = lo // CH, (hi - 1) // CH

    def one(k):
        try:
            raw = np.frombuffer(fetch(B + prefix + '%d/%d/%d/%d' % ((lv,) + k)), np.uint8)
        except Exception:
            return
        a = raw.reshape(CH, CH, CH)
        base = np.array(k) * CH
        s0 = np.maximum(lo, base)
        s1 = np.minimum(hi, base + CH)
        out[tuple(slice(a_, b_) for a_, b_ in zip(s0 - lo, s1 - lo))] = \
            a[tuple(slice(a_, b_) for a_, b_ in zip(s0 - base, s1 - base))]
    keys = [(a, b, c) for a in range(c0[0], c1[0] + 1) for b in range(c0[1], c1[1] + 1) for c in range(c0[2], c1[2] + 1)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, keys))
    return out


def to_pitch(vol, um_from, um_to):
    t = torch.from_numpy(vol).to(dev, torch.float32)[None, None]
    size = [max(1, int(round(s * um_from / um_to))) for s in vol.shape]
    return F.interpolate(t, size=size, mode='trilinear', align_corners=False)[0, 0]


def rot_xy(v, deg, mirror):
    if mirror:
        v = torch.flip(v, dims=[2])
    th = np.deg2rad(deg)
    A = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0]], dtype=torch.float32, device=dev)
    g = F.affine_grid(A[None].expand(v.shape[0], 2, 3), (v.shape[0], 1, v.shape[1], v.shape[2]), align_corners=False)
    return F.grid_sample(v[:, None], g, mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0]


class Big:
    """The coarse volume's FFT terms. CENTRED and scaled to unit sd first: the first run of this script
    (2026-10-05 10:39) scored NCC 4.73, which no NCC can reach - float32 cancellation in s2 - s1^2/n
    where the slab overlapped near-empty volume. NCC is invariant to that shift and scale."""
    def __init__(self, big, tshape, pad):
        nz = big[big > 0]
        self.v0 = float(nz.var() / big.var()) if nz.numel() else 1.0   # papyrus variance, in units of the whole
        big = (big - big.mean()) / big.std()
        self.big, self.pad = big, pad
        self.fsz = (big.shape[0], big.shape[1] + pad, big.shape[2] + pad)
        self.Fb = torch.fft.rfftn(big, s=self.fsz)
        self.Fb2 = torch.fft.rfftn(big * big, s=self.fsz)
        self.zmax = big.shape[0] - tshape[0]


VAR_FRAC = 0.25   # an overlap carrying under a quarter of the scroll's own variance is empty volume, not a match


def ncc_search(B_, tmpl, limit=None):
    m = (tmpl > 0).float()
    n = m.sum()
    t = (tmpl - (tmpl * m).sum() / n) * m
    fsz = B_.fsz
    Ft = torch.conj(torch.fft.rfftn(t, s=fsz))
    Fm = torch.conj(torch.fft.rfftn(m, s=fsz))
    num = torch.fft.irfftn(B_.Fb * Ft, s=fsz)
    s1 = torch.fft.irfftn(B_.Fb * Fm, s=fsz)
    var = torch.fft.irfftn(B_.Fb2 * Fm, s=fsz)
    var.sub_(s1 * s1 / n)
    del s1, Ft, Fm
    floor = VAR_FRAC * B_.v0 * float(n)
    empty = var < floor
    var.clamp_(min=floor)
    score = num / (torch.sqrt(var) * torch.sqrt((t * t).sum()))
    score[empty] = -1
    del num, var, empty
    score[B_.zmax + 1:] = -1
    Y, X = fsz[1], fsz[2]
    if limit is not None:
        yy = torch.arange(Y, device=dev); yy = torch.minimum(yy, Y - yy)
        xx = torch.arange(X, device=dev); xx = torch.minimum(xx, X - xx)
        score[:, yy > limit, :] = -1
        score[:, :, xx > limit] = -1
    i = int(torch.argmax(score))
    best = float(score.flatten()[i])
    del score
    dz, rem = divmod(i, Y * X)
    dy, dx = divmod(rem, X)
    dy = dy if dy <= Y // 2 else dy - Y
    dx = dx if dx <= X // 2 else dx - X
    return best, dz, dy, dx


def place(v, shape, dy, dx):
    out = torch.zeros((v.shape[0],) + tuple(shape), device=dev)
    h, w = min(v.shape[1], shape[0]), min(v.shape[2], shape[1])
    out[:, :h, :w] = v[:, :h, :w]
    return torch.roll(out, (dy, dx), (1, 2))


def ncc_shift(a, b, r):
    a = torch.from_numpy(a).to(dev, torch.float32)
    b = torch.from_numpy(b).to(dev, torch.float32)
    m = (a > 0).float()
    n = m.sum()
    if n < 0.3 * a.numel():
        return None
    t = (a - (a * m).sum() / n) * m
    fsz = b.shape
    Fb = torch.fft.rfftn(b, s=fsz)
    Ft = torch.conj(torch.fft.rfftn(t, s=fsz))
    Fm = torch.conj(torch.fft.rfftn(m, s=fsz))
    num = torch.fft.irfftn(Fb * Ft, s=fsz)
    s1 = torch.fft.irfftn(Fb * Fm, s=fsz)
    s2 = torch.fft.irfftn(torch.fft.rfftn(b * b, s=fsz) * Fm, s=fsz)
    var = (s2 - s1 * s1 / n).clamp_min(1e-3)
    sc = (num / (torch.sqrt(var) * torch.sqrt((t * t).sum())))[:2 * r + 1, :2 * r + 1, :2 * r + 1]
    i = int(torch.argmax(sc))
    d = np.array(np.unravel_index(i, sc.shape)) - r
    return float(sc.flatten()[i]), d


def resample(v, scale, deg):
    t = torch.from_numpy(v).to(dev, torch.float32)[None, None]
    size = [int(round(s * scale)) for s in v.shape]
    t = F.interpolate(t, size=size, mode='trilinear', align_corners=False)[0, 0]
    th = np.deg2rad(deg)
    A = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0]], dtype=torch.float32, device=dev)
    g = F.affine_grid(A[None].expand(t.shape[0], 2, 3), (t.shape[0], 1, t.shape[1], t.shape[2]), align_corners=False)
    return F.grid_sample(t[:, None], g, mode='bilinear', align_corners=False)[:, 0].cpu().numpy()


def u8(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


os.makedirs(WORK, exist_ok=True)
say('%s: the %.3f um scan (%s) into the %.3f um frame (%s); device %s' % (SAMPLE, FINE_UM, FINE_ID, COARSE_UM, COARSE_ID, dev))
fine = level(fine_p, FINE_LV)
coarse = level(coarse_p, COARSE_LV)
if FETCH_ONLY:
    say('  fetch only: levels cached under %s' % WORK)
    runfile.done(OUT, '\n'.join(lines))
    raise SystemExit(0)

# ---- stage 1: every 2 deg at ~300 um, both hands ------------------------------------------------
P1 = FINE_UM * 2 ** FINE_LV * 4
f1 = to_pitch(fine, FINE_UM * 2 ** FINE_LV, P1)
c1 = to_pitch(coarse, COARSE_UM * 2 ** COARSE_LV, P1)
say('  stage 1 at %.1f um: fine slab %s, coarse %s' % (P1, tuple(f1.shape), tuple(c1.shape)))
B1 = Big(c1, f1.shape, max(f1.shape[1], f1.shape[2]))
rows = []
for mirror in (False, True):
    for deg in range(0, 360, 2):
        r = rot_xy(f1, deg, mirror)
        s, dz, dy, dx = ncc_search(B1, place(r, c1.shape[1:], 0, 0))
        rows.append((s, deg, mirror, dz, dy, dx))
rows.sort(reverse=True)
best = rows[0]
say('')
say('  stage 1 top angles, normalised cross-correlation over the slab:')
for s, deg, mir, dz, dy, dx in rows[:8]:
    say('    %6.4f  %3d deg  %-6s  dz %4d  dy %4d  dx %4d' % (s, deg, 'mirror' if mir else 'plain', dz, dy, dx))
rival = next((r for r in rows[1:] if min(abs(r[1] - best[1]), 360 - abs(r[1] - best[1])) > 5 or r[2] != best[2]), None)
other_hand = max(r[0] for r in rows if r[2] != best[2])
say('  best %.4f; best rival more than 5 deg away or other hand %.4f (%s %d deg); best of the other hand %.4f; median %.4f'
    % (best[0], rival[0], 'mirror' if rival[2] else 'plain', rival[1], other_hand, float(np.median([r[0] for r in rows]))))
_cat = json.load(open('geom/metadata-new.json', encoding='utf-8'))['samples'][SAMPLE]['sample']['properties']['volume_transforms']
_Mc = [np.array(t['matrix']) for e in _cat if e['from_volume_id'] == FINE_ID for t in e['transforms'] if t['to_volume_id'] == COARSE_ID]
if _Mc:   # a check printed after the search, never an input to it
    say('  (catalog matrix puts the fine slab\'s slice 0 at coarse z %.0f at this pitch; the search said %d)'
        % (_Mc[0][2, 3] * COARSE_UM / P1, best[3]))
if best[0] > 1.0001:
    runfile.done(OUT, '\n'.join(lines + ['  an NCC above 1 is arithmetic, not a match. Stopped.']))
    raise SystemExit(1)
del B1, f1, c1
if dev == 'cuda':
    torch.cuda.empty_cache()

# ---- stage 2: +-3 deg in quarters at ~150 um ------------------------------------------------------
PITCH = P1 / 2
f = to_pitch(fine, FINE_UM * 2 ** FINE_LV, PITCH)
c = to_pitch(coarse, COARSE_UM * 2 ** COARSE_LV, PITCH)
z0 = max(0, best[3] * 2 - 12)
cz = c[z0:z0 + f.shape[0] + 24]
B2 = Big(cz, f.shape, 32)
fine_rows = []
for k in range(-12, 13):
    deg = best[1] + k * 0.25
    r = place(rot_xy(f, deg, best[2]), cz.shape[1:], best[4] * 2, best[5] * 2)
    s, dz, dy, dx = ncc_search(B2, r, limit=12)
    fine_rows.append((s, deg, best[2], z0 + dz, best[4] * 2 + dy, best[5] * 2 + dx))
fine_rows.sort(reverse=True)
say('')
say('  stage 2 at %.1f um, +-3 deg in quarter degrees:' % PITCH)
for s, deg, mir, dz, dy, dx in fine_rows[:5]:
    say('    %6.4f  %6.2f deg  dz %4d  dy %4d  dx %4d' % (s, deg, dz, dy, dx))
curve = sorted(fine_rows, key=lambda r: r[1])
say('  score by angle: ' + ' '.join('%.3f' % r[0] for r in curve))
best = fine_rows[0]
del B2
s, deg, mir, dz, dy, dx = best
fr = place(rot_xy(f, deg, mir), c.shape[1:], dy, dx)
zm = fr.shape[0] // 2
a = c[dz + zm].cpu().numpy()
b = fr[zm].cpu().numpy()
h = max(a.shape[0], b.shape[0])
pic = np.zeros((h, a.shape[1] + b.shape[1] + 10), np.uint8)
pic[:a.shape[0], :a.shape[1]] = u8(a)
pic[:b.shape[0], a.shape[1] + 10:] = u8(b)
Image.fromarray(pic).save(os.path.join(WORK, 'overlay.png'))
cj = {'sample': SAMPLE, 'pitch_um': PITCH, 'deg': deg, 'mirror': mir, 'dz': dz, 'dy': dy, 'dx': dx,
      'score': s, 'stage1_rival': rival[0], 'stage1_other_hand': other_hand}
json.dump(cj, open(os.path.join(WORK, 'coarse.json'), 'w'), indent=1)
say('  wrote %s/overlay.png (coarse slice | fine slice, registered) and coarse.json' % WORK)
del f, c, cz, fr
if dev == 'cuda':
    torch.cuda.empty_cache()
if cj['mirror']:
    runfile.done(OUT, '\n'.join(lines + ['  stage 1 says mirror; the block refinement below assumes plain. Stopped.']))
    raise SystemExit(1)

# ---- stage 3: blocks at full 9.366 um, an affine, leave-one-out ----------------------------------
fz = json.loads(fetch(B + fine_p + '0/.zarray'))['shape']
P = cj['pitch_um']
fine_c = np.array([0, (fz[1] - 1) / 2, (fz[2] - 1) / 2])
kf, kc = FINE_UM / P, COARSE_UM / P
th = np.deg2rad(cj['deg'])
R = np.array([[1, 0, 0], [0, np.cos(th), np.sin(th)], [0, -np.sin(th), np.cos(th)]])


def fine_to_coarse0(pf):
    q = (pf - fine_c) * kf
    q = np.linalg.inv(R) @ q
    q = q + np.array([0, (fz[1] - 1) / 2 * kf, (fz[2] - 1) / 2 * kf]) + np.array([cj['dz'], cj['dy'], cj['dx']])
    return q / kc


ov = fine
cands = []
for z in np.linspace(0.12, 0.88, 4):
    for y in np.linspace(0.3, 0.7, 3):
        for x in np.linspace(0.3, 0.7, 3):
            p5 = np.array([z * ov.shape[0], y * ov.shape[1], x * ov.shape[2]]).astype(int)
            if ov[tuple(p5)] > 0 and ov[p5[0], p5[1] - 2:p5[1] + 3, p5[2] - 2:p5[2] + 3].mean() > 20:
                cands.append(p5 * 32)
n_grid = len(cands)
# and a 2x2x2 grid inside the control mesh's bounding box, where the map is used
CX, CY, CZ = (np.array(Image.open(os.path.join(CTRL, k + '.tif')), np.float64) for k in ('x', 'y', 'z'))
cv = (CX > 0) & (CY > 0) & (CZ > 0)
lo = np.array([CZ[cv].min(), CY[cv].min(), CX[cv].min()])
hi = np.array([CZ[cv].max(), CY[cv].max(), CX[cv].max()])
for fz_ in (0.25, 0.75):
    for fy_ in (0.25, 0.75):
        for fx_ in (0.25, 0.75):
            p0 = lo + (hi - lo) * np.array([fz_, fy_, fx_])
            p5 = (p0 / 32).astype(int)
            if ov[p5[0], p5[1] - 2:p5[1] + 3, p5[2] - 2:p5[2] + 3].mean() > 20:
                cands.append(p5 * 32)
say('')
say('  stage 3: %d block centres with papyrus under them (%d on the slab grid of 36, %d of 8 in the control box)'
    % (len(cands), n_grid, len(cands) - n_grid))

pairs = []
ratio = COARSE_UM / (FINE_UM * 4)
for pf in cands:
    pc = fine_to_coarse0(pf.astype(float))
    c_lo = np.round(pc).astype(int) - HALF - SEARCH
    c_hi = c_lo + 2 * (HALF + SEARCH)
    f2 = pf / 4.0
    fh = int(np.ceil((HALF + 8) * ratio))
    f_lo = np.round(f2).astype(int) - fh
    f_hi = f_lo + 2 * fh
    if (f_lo < 0).any() or (c_lo < 0).any():
        continue
    cb = box(coarse_p, 0, c_lo, c_hi)
    fb = box(fine_p, 2, f_lo, f_hi)
    frs = resample(fb, (FINE_UM * 4) / COARSE_UM, cj['deg'])
    m0 = (np.array(frs.shape) - 2 * HALF) // 2
    a = frs[m0[0]:m0[0] + 2 * HALF, m0[1]:m0[1] + 2 * HALF, m0[2]:m0[2] + 2 * HALF]
    got = ncc_shift(a, cb, SEARCH)
    if got is None:
        say('  block at fine %s: under 30%% papyrus, skipped' % (pf.tolist(),))
        continue
    sc, d = got
    pf_true = (f_lo + fh) * 4.0
    pc_true = c_lo + SEARCH + d + HALF
    pairs.append((pf_true, pc_true, sc, any((pf == q).all() for q in cands[n_grid:])))
    say('  block at fine %-22s NCC %.3f  shift %s coarse voxels%s' % (
        pf_true.astype(int).tolist(), sc, d.tolist(), '  (control box)' if pairs[-1][3] else ''))

keep = [p for p in pairs if p[2] >= 0.5]
say('')
say('  %d blocks matched, %d at NCC >= 0.5 kept for the fit' % (len(pairs), len(keep)))
if len(keep) < 6:
    runfile.done(OUT, '\n'.join(lines + ['  too few blocks to fit an affine; not written']))
    raise SystemExit(1)
X = np.array([np.append(p[0], 1) for p in keep])
Y = np.array([p[1] for p in keep])
A, *_ = np.linalg.lstsq(X, Y, rcond=None)              # coarse = [fine, 1] @ A
res = np.linalg.norm(X @ A - Y, axis=1) * COARSE_UM
loo = []
for i in range(len(keep)):
    m = np.arange(len(keep)) != i
    Ai, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
    loo.append(np.linalg.norm(X[i] @ Ai - Y[i]) * COARSE_UM)
loo = np.array(loo)
box_i = np.array([p[3] for p in keep])
M = A[:3].T
sv = np.linalg.svd(M, compute_uv=False)
say('  affine fit: scale per axis (singular values / %.5f nominal) %s; det sign %+d'
    % (FINE_UM / COARSE_UM, ' '.join('%.5f' % (s_ / (FINE_UM / COARSE_UM)) for s_ in sv), int(np.sign(np.linalg.det(M)))))
say('  residual per block, um:          ' + ' '.join('%.0f' % r for r in res))
say('  leave-one-out residual, um:      ' + ' '.join('%.0f' % r for r in loo))
say('  median %.1f um in-sample, %.1f um held out (max held out %.1f um)' % (np.median(res), np.median(loo), loo.max()))
if box_i.any():
    say('  held out, the control-box blocks only: median %.1f um, max %.1f um (n %d)'
        % (np.median(loo[box_i]), loo[box_i].max(), int(box_i.sum())))

# ---- the catalog's own matrix, never fitted to: how far apart are the two maps? --------------------
cat = json.load(open('geom/metadata-new.json', encoding='utf-8'))['samples'][SAMPLE]['sample']['properties']['volume_transforms']
Mc = None
for e in cat:
    if e['from_volume_id'] == FINE_ID:
        for t in e['transforms']:
            if t['to_volume_id'] == COARSE_ID:
                Mc = np.array(t['matrix'])
say('')
if Mc is None:
    say('  catalog: no %s -> %s matrix' % (FINE_ID, COARSE_ID))
    cat_ctrl = None
else:
    def cat_map(zyx):
        xyz = zyx[..., ::-1]                               # catalog matrices act on (x, y, z)
        out = xyz @ Mc[:, :3].T + Mc[:, 3]
        return out[..., ::-1]
    Pk = np.array([p[0] for p in keep])
    dcat = np.linalg.norm(cat_map(Pk) - Y, axis=1) * COARSE_UM
    say('  catalog matrix (x,y,z order) on our matched blocks: median %.1f um, max %.1f um from where NCC put them'
        % (np.median(dcat), dcat.max()))
    Pc = np.stack([CZ[cv], CY[cv], CX[cv]], -1)
    ours = np.append(Pc, np.ones((len(Pc), 1)), 1) @ A
    dctl = np.linalg.norm(ours - cat_map(Pc), axis=1) * COARSE_UM
    cat_ctrl = (float(np.median(dctl)), float(np.percentile(dctl, 95)), float(dctl.max()))
    say('  our affine vs the catalog matrix on the %d control-mesh vertices: median %.1f um, p95 %.1f um, max %.1f um'
        % (len(Pc), *cat_ctrl))

# side check: is level 2 the mean of level 1?
pf = cands[len(cands) // 2]
l1 = box(fine_p, 1, (pf // 2 - 32).astype(int), (pf // 2 + 32).astype(int)).astype(np.float32)
l2 = box(fine_p, 2, (pf // 4 - 16).astype(int), (pf // 4 + 16).astype(int)).astype(np.float32)
mean2 = l1.reshape(32, 2, 32, 2, 32, 2).mean(axis=(1, 3, 5))
stride2 = l1[::2, ::2, ::2]
say('  level 2 against level 1 pooled 2x: mean |diff| %.2f; against level 1 strided: %.2f  (grey levels)'
    % (float(np.abs(l2 - mean2).mean()), float(np.abs(l2 - stride2).mean())))
json.dump({'sample': SAMPLE, 'fine_volume': FINE_ID, 'coarse_volume': COARSE_ID, 'fine_to_coarse_affine': A.tolist(),
           'form': 'coarse_voxel(z,y,x) = [fine_voxel(z,y,x), 1] @ A, both at level 0',
           'median_residual_um': float(np.median(res)), 'median_loo_um': float(np.median(loo)),
           'max_loo_um': float(loo.max()), 'blocks': len(keep),
           'control_box_loo_um': [float(v) for v in loo[box_i]],
           'vs_catalog_on_control_mesh_um_median_p95_max': cat_ctrl, 'coarse': cj},
          open(AFF_OUT, 'w'), indent=1)
say('  wrote %s   %.0f s' % (AFF_OUT, time.time() - t0))
runfile.done(OUT, '\n'.join(lines))
