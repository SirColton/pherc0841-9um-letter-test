"""day1-gate1-full.py - prereg v2 sanity gate 1, in full: the model-free direct render of 252405 (layers
18-22 mean of render/252405-stack.npy) against ScrollPrize's 2.4 um map, by raw r AND by hp r at blur 0,
each against its 48-shift rolled-key null. Day 1 ran only the raw half (day1-render-9um.py).

WHY. If the render with no model in it already tracks the 2.4 um ink map beyond chance, density or
geometry rather than ink can carry a 9 um model's score, and the decisive numbers mean less.

DEFINITIONS, from the scripts the prereg cites:
  pixels   render mask with a 40 px edge cut (day1-render-9um.py gate 1), 4,468,325 px.
  key      render/252405-spmap-grid.npy (ScrollPrize's full-res 2.4 um map by index), never blurred.
  raw r    Pearson over the pixels; null = the key shifted by c0841's 48 shifts (104 px steps, 1-3 mm per axis).
  hp r     day1-score-baselines.py's: hp(x) = x - G(x m)/G(m) on the pixels m, core = m where G(m, 2 sigma)
           > 0.999, Pearson over the core, read not blurred (blur 0); null = the hp key with its core rolled.
  nulls    both constructions are run: WRAP (np.roll of key and mask, day1-score-baselines.py) and OVERLAP
           (shifted crop, day1-render-9um.py). Day 1's raw number is reproduced by the overlap one.
CONSERVATIVE READINGS, marked because the prereg is silent on them:
  - sigma: hp r is run at 48 um and at 192 um (the prereg's two reported bands); both must be inside.
  - inside: null min <= r <= null max (two-sided), in BOTH null constructions. Day 1's one-sided rule
    (r <= null max) is printed beside.
  - blur 2 at 192 um (the decisive metric's blur) is printed beside, outside the verdict.
w00big (c0841 frame, SP ds8 key, day1-score-baselines.py pixels) is printed as a reference, not gated.
Usage: day1-gate1-full.py <out txt>     CPU only, no model; writes <out txt> and <out>.json. Run from <workdir>.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

import runfile

OUT = sys.argv[1]
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


PX = 2.403 * 4
E = 40
SHIFTS = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
SIGMAS = (48.0, 192.0)


def pr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])


def shifted(a, b, m, dy, dx):
    H, W = a.shape
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = m[ys:ye, xs:xe] & m[ys - dy:ye - dy, xs - dx:xe - dx]
    if mm.sum() < 1000:
        return np.nan
    return float(np.corrcoef(a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm])[0, 1])


def summary(r0, nl):
    nl = np.asarray([v for v in nl if np.isfinite(v)])
    return {'r': r0, 'n_null': int(len(nl)), 'null_mean': float(nl.mean()), 'null_sd': float(nl.std()), 'null_min': float(nl.min()),
            'null_max': float(nl.max()), 'z': float((r0 - nl.mean()) / max(nl.std(), 1e-12)), 'n_ge': int((nl >= r0).sum()),
            'n_le': int((nl <= r0).sum()), 'inside_two_sided': bool(nl.min() <= r0 <= nl.max()), 'inside_one_sided': bool(r0 <= nl.max())}


def gate(direct, key, M, VAL, blurs_hp=((48.0, 0), (192.0, 0), (192.0, 2))):
    res = {}
    r0 = pr(direct, key, M)
    res['raw b0'] = {'wrap': summary(r0, [pr(direct, np.roll(key, s, (0, 1)), M & np.roll(M, s, (0, 1))) for s in SHIFTS]),
                     'overlap': summary(r0, [shifted(direct, key, M, dy, dx) for dy, dx in SHIFTS])}
    m = M.astype(np.float32)
    for sig_um, blur in blurs_hp:
        sig = sig_um / PX
        hp = lambda x: x - gaussian_filter(x * m, sig) / np.maximum(gaussian_filter(m, sig), 1e-3)
        core = M & (gaussian_filter(m, 2 * sig) > 0.999)
        x = direct
        if blur:
            w = gaussian_filter(VAL.astype(np.float32), blur)        # day1-score-baselines.py mblur: weights on the uncut mask
            x = gaussian_filter(np.where(VAL, direct, 0).astype(np.float32), blur) / np.maximum(w, 1e-3)
        xh, kh = hp(x), hp(key)
        r0 = pr(xh, kh, core)
        res['hp%d b%d' % (sig_um, blur)] = {
            'n_core': int(core.sum()),
            'wrap': summary(r0, [pr(xh, np.roll(kh, s, (0, 1)), core & np.roll(core, s, (0, 1))) for s in SHIFTS]),
            'overlap': summary(r0, [shifted(xh, kh, core, dy, dx) for dy, dx in SHIFTS])}
    return res


def show(name, res, gated):
    for k, v in res.items():
        for nk in ('wrap', 'overlap'):
            s = v[nk]
            say('    %-8s %-7s r %+.4f | null (%d) mean %+.4f sd %.4f min %+.4f max %+.4f | z %+5.1f  n>=r %2d  n<=r %2d | %s%s'
                % (k, nk, s['r'], s['n_null'], s['null_mean'], s['null_sd'], s['null_min'], s['null_max'], s['z'], s['n_ge'], s['n_le'],
                   'INSIDE' if s['inside_two_sided'] else 'OUTSIDE', '' if k in gated else '   (beside, not gated)'))


say('day1-gate1-full.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
R = 'geom/day1-9um/render/'
stack = np.load(R + '252405-stack.npy', mmap_mode='r')
grid = json.load(open(R + '252405-grid.json'))
c = grid['centre_layer']
assert (c - 2, c + 2) == (18, 22), c
direct = np.asarray(stack[c - 2:c + 3], np.float32).mean(0)
mask = np.load(R + '252405-mask.npy')
key = np.load(R + '252405-spmap-grid.npy').astype(np.float32) * 255
H, W = mask.shape
M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
say('  252405: stack %s (sha1 %s per grid.json), direct render = layers %d-%d mean; key = render/252405-spmap-grid.npy; pixels %d (%.2f cm2)'
    % (stack.shape, grid['stack_sha1'], c - 2, c + 2, M.sum(), M.sum() * (PX * 1e-4) ** 2))
say('  nulls: 48 shifts, 104 px (1.0 mm) steps to 312 px per axis; WRAP = np.roll of key and mask; OVERLAP = shifted crop')
say('  hp sigma 48 um = %.2f px, 192 um = %.2f px; blur 0 (read not smoothed), key never smoothed' % (48 / PX, 192 / PX))
say('  day-1 raw r on record (render/252405-gate1.json, overlap null): r +0.0516, null max +0.0718')
say('')
GATED = ('raw b0', 'hp48 b0', 'hp192 b0')
res = gate(direct, key, M, mask)
show('252405', res, GATED)
ok = all(res[k][nk]['inside_two_sided'] for k in GATED for nk in ('wrap', 'overlap'))
narrow = res['raw b0']['overlap']['inside_one_sided'] and res['hp192 b0']['overlap']['inside_one_sided']
say('')
say('  GATE 1 (conservative: raw, hp 48 um and hp 192 um at blur 0, two-sided, both null constructions): %s' % ('PASS' if ok else 'FAIL'))
for k in GATED:
    say('    %-8s wrap %s, overlap %s' % (k, 'inside' if res[k]['wrap']['inside_two_sided'] else 'OUTSIDE',
                                         'inside' if res[k]['overlap']['inside_two_sided'] else 'OUTSIDE'))
say('  beside, the narrowest reading (day 1 rule r <= null max, overlap null, raw and hp 192 um only): %s' % ('PASS' if narrow else 'FAIL'))

# ---- w00big reference (not gated) ----------------------------------------------------------------
say('')
say('  REFERENCE w00big (selection frame; day1-score-baselines.py pixels and SP ds8 key; not gated):')
B = 'geom/l0-1203-render/audit/build-deciding-test'
CT = 'geom/c0841/mesh0b-render/w00big'
Hw, Ww, P = 995, 1245, 4
st = np.load(os.path.join(B, 'ours-stack.npy'), mmap_mode='r')
dw = np.asarray(st[18:23], np.float32).mean(0)
mw = np.load(os.path.join(B, 'ours-mask.npy'))
m24 = np.asarray(Image.open(os.path.join(CT, 'mask.png')), np.float32)[:Hw * P, :Ww * P].reshape(Hw, P, Ww, P).min((1, 3)) > 0
Mw = mw & m24; Mw[:E] = Mw[-E:] = False; Mw[:, :E] = Mw[:, -E:] = False
ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
gi = 2.5 * (380 + (np.arange(Hw) * P + (P - 1) / 2) / 20.0) - 0.5
gj = 2.5 * (500 + (np.arange(Ww) * P + (P - 1) / 2) / 20.0) - 0.5
GJ, GI = np.meshgrid(gj, gi)
kw = map_coordinates(ink, [GI, GJ], order=1).astype(np.float32)
say('    pixels %d; stack %s' % (Mw.sum(), st.shape))
resw = gate(dw, kw, Mw, mw & m24)
show('w00big', resw, ())

json.dump({'prereg_sha256_expected': '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c', 'n_pixels': int(M.sum()),
           'gated': list(GATED), 'pass_conservative': bool(ok), 'pass_narrow_day1_rule': bool(narrow), '252405': res, 'w00big_reference': resw},
          open(os.path.splitext(OUT)[0] + '.json', 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
