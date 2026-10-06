"""day1-score-baselines.py - score every ink_9um-format map on the w00big native 9.366 um frame.

Plan step 2, on-disk models only. Every map is scored on identical pixels, two ways:
  AUC against ScrollPrize's w00 ink labels (inklabels_v2 carried onto the grid by
    labels-on-grid.py): positives label >= 0.75; negatives label <= 0 AND supervision_mask_v2 >= 0.75
    (the public 0841 convention, auc-in-supervision.py); AUC with all label-0 negatives beside it.
  hp r, Scheirer's letter-scale score (vesuvius-reports 02-9um-ink-reader-benchmark hp_score.py):
    hp(x) = x - G(x*m)/G(m), Gaussian sigma 48 um (his SIG = 20/3.9 px on a 9.36 um grid; here
    48/9.61 px), core = mask eroded where G(mask, 2 sigma) > 0.999, Pearson over the core, on both
    read and key. Key = ScrollPrize's w00 2.4 um ink map (the ds8 jpg on disk, 19.2 um px, bilinear
    onto the grid as c0841-9um.py does). Read blurred by 0, 2, 4 px (Gaussian, masked) before the
    high-pass; the key never is. Null = the hp key rolled by 48 shifts of 1-3 mm per axis
    (c0841-9um.py's set, 104-px steps), with the core rolled with it.
  Second key, for the ds8 key's resolution: our 2.4 um Hecate rev map of the w00big render, 4x4
    pooled (full 2.4 um resolution, r 0.82 with the SP map), hp r at blur 2 with the same null.
Maps: <maps>/<model>-c<+d>-<order>.npy from day1-ink9-maps.py; per (model, order) also the
five-centre mean (-4,-2,0,+2,+4) and the three-centre mean (-4,0,+4) where present.
CPU only. Usage: day1-score-baselines.py <maps dir> <out prefix> [--sigma <um>, default 48]   (writes <out>.txt, <out>.json)
"""
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import rankdata

import runfile

ARGS = sys.argv[1:]
SIG_UM = 48.0                        # --sigma <um>: the hp Gaussian; 48 is Scheirer's letter scale
if '--sigma' in ARGS:
    _i = ARGS.index('--sigma'); SIG_UM = float(ARGS[_i + 1]); del ARGS[_i:_i + 2]
MAPS, OUTP = ARGS[0], ARGS[1]
runfile.claim(OUTP + '.txt')
V = 'geom/l0-1203-render/audit/verify-nine-um-test-does-it-matter'
B = 'geom/l0-1203-render/audit/build-deciding-test'
CTRL24 = 'geom/c0841/mesh0b-render/w00big'
H, W, P, E = 995, 1245, 4, 40
PX = 2.403 * P                       # 9.612 um grid
SIG = SIG_UM / PX
BLURS = (0, 2, 4)
SHIFTS = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


lab = np.load(os.path.join(V, 'w00_inklabels_v2-grid.npy'))
sup = np.load(os.path.join(V, 'w00_supervision_mask_v2-grid.npy'))
mask = np.load(os.path.join(B, 'ours-mask.npy'))
m24 = np.asarray(Image.open(os.path.join(CTRL24, 'mask.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
VAL = mask & m24
M = VAL.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
core = M & (gaussian_filter(M.astype(np.float32), 2 * SIG) > 0.999)
pos = M & (lab >= 0.75); negA = M & (lab <= 0); negS = negA & (sup >= 0.75)

ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
gi = 2.5 * (380 + (np.arange(H) * P + (P - 1) / 2) / 20.0) - 0.5
gj = 2.5 * (500 + (np.arange(W) * P + (P - 1) / 2) / 20.0) - 0.5
GJ, GI = np.meshgrid(gj, gi)
spink = map_coordinates(ink, [GI, GJ], order=1).astype(np.float32)
h24 = np.asarray(Image.open(os.path.join(CTRL24, 'hecate-rev.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3))


def auc(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]).astype(np.float64)
    rk = rankdata(v)                 # midranks: the maps are uint8 means, so ties are many
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mblur(x, s):
    if not s:
        return x
    w = gaussian_filter(VAL.astype(np.float32), s)
    return gaussian_filter(np.where(VAL, x, 0).astype(np.float32), s) / np.maximum(w, 1e-3)


def hp(x):
    m = M.astype(np.float32)
    return x - gaussian_filter(x * m, SIG) / np.maximum(gaussian_filter(m, SIG), 1e-3)


def pr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])


KEYS = {'sp': hp(spink), 'h24': hp(h24)}
NULLK = {k: [(np.roll(kh, s, (0, 1)), core & np.roll(core, s, (0, 1))) for s in SHIFTS] for k, kh in KEYS.items()}


def hpr(x, key, blur):
    xh = hp(mblur(np.asarray(x, np.float32), blur))
    r0 = pr(xh, KEYS[key], core)
    nl = np.array([pr(xh, kr, mm) for kr, mm in NULLK[key]])
    return {'r': r0, 'null_mean': float(nl.mean()), 'null_max': float(nl.max()), 'null_absmax': float(np.abs(nl).max()),
            'z': float((r0 - nl.mean()) / max(nl.std(), 1e-9)), 'n_null_ge': int((nl >= r0).sum())}


maps = {}
found = {}
for f in sorted(glob.glob(os.path.join(MAPS, '*.npy'))):
    m = re.match(r'(.+)-c([+-]\d+)-(rev|fwd)\.npy$', os.path.basename(f))
    if m:
        found.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = f
for (mod, o), d in sorted(found.items()):
    for c in sorted(d):
        maps['%s %s c%+d' % (mod, o, c)] = np.load(d[c]).astype(np.float32)
    if all(c in d for c in (-4, 0, 4)):
        maps['%s %s mean3(-4,0,+4)' % (mod, o)] = np.mean([maps['%s %s c%+d' % (mod, o, c)] for c in (-4, 0, 4)], 0)
    if all(c in d for c in (-4, -2, 0, 2, 4)):
        maps['%s %s mean5' % (mod, o)] = np.mean([maps['%s %s c%+d' % (mod, o, c)] for c in (-4, -2, 0, 2, 4)], 0)


def rnorm(a):
    r = np.full(a.shape, 0.5, np.float32)
    r[M] = rankdata(a[M]) / M.sum()
    return r


# extra on-disk recipes, not in the plan's list: rank-normalised prediction means of the mean5 maps
for members in (('s43_75k', 'soup42', 'soup43'), ('soup42', 'soup43')):
    ks = ['%s rev mean5' % m for m in members]
    if all(k in maps for k in ks):
        maps['ENS rank-mean %s rev mean5' % '+'.join(members)] = np.mean([rnorm(maps[k]) for k in ks], 0)
        maps['ENS raw-mean %s rev mean5' % '+'.join(members)] = np.mean([maps[k] for k in ks], 0)
# control for the ensembles: the rank transform alone, one member at a time (AUC cannot move; hp r can)
for m in ('s43_75k', 'soup42', 'soup43'):
    if '%s rev mean5' % m in maps:
        maps['RANK %s rev mean5 (transform only)' % m] = rnorm(maps['%s rev mean5' % m])
stack = np.load(os.path.join(B, 'ours-stack.npy'), mmap_mode='r')
refs = {'REF direct render, no model (layers 18-22)': np.asarray(stack[18:23], np.float32).mean(0),
        'REF ScrollPrize 2.4 um map (ds8)': spink, 'REF Hecate 2.4 um rev (ours)': h24}

say('day1-score-baselines.py  maps from %s' % MAPS)
say('  frame %dx%d at %.2f um; scored px %d; hp core %d; pos %d; neg(sup) %d; neg(all) %d' % (
    H, W, PX, M.sum(), core.sum(), pos.sum(), negS.sum(), negA.sum()))
say('  hp: Gaussian sigma %g um = %.2f px; null 48 rolled-key shifts, 1.0-3.0 mm per axis' % (SIG_UM, SIG))
say('')
hdr = '%-34s %7s %7s | %7s %7s %7s %6s %6s | %7s %7s | %7s %7s | %7s %7s' % (
    'map', 'AUCsup', 'AUCall', 'hp b0', 'nullmx', 'abs mx', 'z', 'n>=r', 'hp b2', 'nullmx', 'hp b4', 'nullmx', 'h24 b2', 'nullmx')
say(hdr)
rows = []
for name, a in list(refs.items()) + list(maps.items()):
    row = {'map': name, 'auc_sup': auc(a, pos, negS), 'auc_all': auc(a, pos, negA)}
    for b in BLURS:
        row['hp_sp_b%d' % b] = hpr(a, 'sp', b)
    row['hp_h24_b2'] = hpr(a, 'h24', 2)
    rows.append(row)
    say('%-34s %7.3f %7.3f | %+7.3f %+7.3f %7.3f %6.1f %6d | %+7.3f %+7.3f | %+7.3f %+7.3f | %+7.3f %+7.3f' % (
        name, row['auc_sup'], row['auc_all'], row['hp_sp_b0']['r'], row['hp_sp_b0']['null_max'], row['hp_sp_b0']['null_absmax'],
        row['hp_sp_b0']['z'], row['hp_sp_b0']['n_null_ge'], row['hp_sp_b2']['r'], row['hp_sp_b2']['null_max'],
        row['hp_sp_b4']['r'], row['hp_sp_b4']['null_max'], row['hp_h24_b2']['r'], row['hp_h24_b2']['null_max']))
say('')
say('columns: AUCsup = negatives restricted to supervised pixels; hp bN = hp r vs the SP 2.4 um map with the read')
say('  blurred N px; nullmx / abs mx = max and max |r| of the 48 rolled-key nulls; z and n>=r for blur 0;')
say('  h24 b2 = hp r vs our full-resolution 2.4 um Hecate rev map, blur 2.')
json.dump({'sigma_um': SIG_UM, 'sigma_px': SIG, 'px_um': PX, 'n_scored': int(M.sum()), 'n_core': int(core.sum()), 'pos': int(pos.sum()),
           'neg_sup': int(negS.sum()), 'neg_all': int(negA.sum()), 'rows': rows}, open(OUTP + '.json', 'w'), indent=1)
runfile.done(OUTP + '.txt', '\n'.join(lines) + '\n')
