"""student2-test-score.py - P5 of prereg-student-v2.json (sha256 f348e4b8...33d7): student v2 and its twin on 252405,
the SECOND LOOK at that segment. A copy of student-test-score.py (f798e0d0...) whose only edits are those prereg-v2
6_evaluation.test_P5 names:
  the v2 hashes  the FROZEN list pins the v2 prereg, the v2 harness, both v2 final checkpoints, the v1 student and v1 twin
                 exports, and v1's scorer beside v1's pins; the twin null is read from the v2 training logs.
  B per gate     (prereg-v2 2_second_look_gates.baseline_B_per_gate) G1 B = student v1 (v1's exported map,
                 student-rev-mean5.npy 11082328...); G2, G4 and G5 B = TAUIL ft6k (5312246b...).
  1.25th pct     G1 and G2 pass on point >= 0.5 AND the 1.25th bootstrap percentile >= 0.25 (Bonferroni over two looks).
  undefined r    (G4_undefined_r) a control whose hp read has zero variance on a core (r not finite) is reported as
                 'undefined (no variance)' and counted INSIDE, and named.
  reproduction   the v1 student map must rescore to AUCsup 0.8252 and hp192 b2 +0.4543 at 4 dp, beside the TAUIL check
                 (0.8102, +0.3499), or scoring stops.
  v1 rows        the v1 student and the v1 twin are scored as rows on the same pixels; reported, never deciding: the
                 closure against TAUIL with v1's 2.5th-percentile rule, S - B 95% against both baselines, the SP-key
                 literal hp closure against both.
  second look    every verdict line says 'second look at 252405'.
  [machine] the export names are student2-rev-mean5, scramble2-rev-mean5 and depth-null-student2-shuf21-rev-mean5, in
  geom/student-v2/test; when S is byte-equal to the v1 student map (the dry run on staged v1 maps) every G1 number is
  labelled meaningless (S = B) and the verdict line says DRY RUN.
Usage: student2-test-score.py <out.txt> <maps dir> [<maps dir> ...], maps named student2 / scramble2 / student2_shuf21 /
student2_shuf41 as student2-test-ink9-maps.py writes them. CPU only, no model. Run from <workdir>.

v1's text follows (its numbers and B are v1's; the edits above override it).
student-test-score.py - P5 of prereg-student.json (sha256 f3ebf0ab...d3710) + amendment 1 (3b0f1bc4...cf10): the
student and the scramble twin on held-out 252405, scored once, every numeric GO gate applied as written.

DAY 1's SCORER, COPIED (day1-test-score.py, sha256 ad385440...): pixels = render mask minus 40 px; HF v2 grids from
day1-9um/test/labels-hf; pos = label >= 0.75; neg = label <= 0 AND supervision >= 0.75; midrank AUC; C = AUCsup of
ScrollPrize's 2.4 um map (render/252405-spmap-grid.npy) recomputed and asserted equal to labels-hf.json; nAUC =
(AUCsup - 0.5) / (C - 0.5). hp(x) = x - G(x m)/max(G(m), 1e-3) on m = the scored px, core = m AND G(m, 2 sigma) >
0.999, read blurred (masked on the render mask) by 0/2/4 px, key never blurred; key = the SP map; null = 48 shifts in
104 px steps to 312 px, WRAP (np.roll) and OVERLAP (shifted crop). S3 20260918 grids beside.

WHAT IS NEW, each from the prereg:
  baseline B   rescored from its frozen map (day1-9um/test/arm-A-R1-tauil_ft6k-rev-mean5.npy, sha256 5312246b...);
               must give AUCsup 0.8102 and hp192 b2 +0.3499 at 4 dp, or scoring stops (5_evaluation_252405.scoring).
  label key    (5.label_key_hp) K = HF v2 inklabels; m_L = scored px AND (supervision >= 0.75 OR label >= 0.75);
               core_L = m_L AND G(m_L, 2 sigma) > 0.999 at sigma 192 um = 19.975 px; hp with m = m_L; read blurred
               2 px (masked), key never; Pearson over core_L; null = the same 48 shifts, wrap and overlap.
  bootstrap    (6_go.bootstrap) 2 mm tiles = 208 x 208 render px anchored at (0, 0); a tile is in a metric's pool when
               it holds >= 1 of that metric's scored px (AUC: a pos or a neg; hp: a core px); 2,000 resamples of the
               pool with replacement, numpy default_rng(0); T, B and S resampled TOGETHER (pixel multiplicities kept);
               closure = (S - B) / (T - B) per resample, T - B <= 0 counts as -inf.
               [machine] one fresh default_rng(0) per pool, drawn as rng.integers(0, n_pool, size=(2000, n_pool));
               the hp maps are high-passed once on the whole frame and the tiles resample their core px; weighted AUC
               and weighted Pearson are exact under pixel multiplicities (checked against the unweighted values at
               k = 1); the 2.5th percentile is numpy's default linear rule, -inf if either neighbour is -inf.
  gates        G1 AUCsup closure: point >= 0.5 AND 2.5th pct >= 0.25.
               G2 hp192 b2 closure on the label key: informative only if T_K - B_K >= 2 x sd of B's 48 rolled-key
                  (wrap) nulls on core_L [machine: np.std, ddof 0, as day 1's summ(); the overlap sd printed beside];
                  if informative, point >= 0.5 AND 2.5th pct >= 0.25.
               G3 student hp192 b2 on the SP key above its null max, wrap AND overlap.
               G4 192 um b2: fwd, shuf21, shuf41 each <= own null max (wrap and overlap); 48 um b2: fwd <= own null max,
                  shuf21 <= max(own null max, 0.0293), shuf41 <= max(own null max, 0.0207) (wrap and overlap each).
               G5 twin nAUC <= B's nAUC (as rescored) + 0.02.
               twin valid null: read from both training logs (|r| <= 0.10 on both segments).
               G6 the letter read: a separate blind-read run, later. This script cannot pass it.
  The SP-key literal closure on hp192 b2 is computed and REPORTED, never deciding (6_go.why_G2_uses_the_label_key).

Usage: student-test-score.py <out.txt> <maps dir> [<maps dir> ...]     CPU only, no model. Run from <workdir>.
Writes <out.txt>, <out>.json and, under geom/student-9um/test/, the exported mean5 maps (npy + raw8 png) and exports.json.
"""
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

import runfile

OUT = sys.argv[1]
MAPDIRS = sys.argv[2:]
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def stop(msg):
    say('STOP: ' + msg)
    runfile.done(OUT, '\n'.join(lines) + '\n')
    sys.exit(2)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


SD = 'geom/student-v2/'
V1 = 'geom/student-9um/'
EXPORT_DIR = os.environ.get('STUDENT_TEST_EXPORT_DIR', SD + 'test')  # dry runs only point this elsewhere
R, T1 = 'geom/day1-9um/render/', 'geom/day1-9um/test/'
FROZEN = [(SD + 'prereg-student-v2.json', 'f348e4b89b19af4a682c65dfb672661296f909183ead703237711e547faa33d7'),
          (V1 + 'prereg-student.json', 'f3ebf0abc3f93a6772ce31f6915d7f5aef2ea4afc72badeb11cbb735216d3710'),
          (V1 + 'prereg-student-amendment-1.txt', '3b0f1bc434decb39e9b46e87ef5913c4bd53eb9c1a649a3ccb93f6d69f82cf10'),
          ('student2-train.py', 'f11f166f985cd5095d083241c692a346b7fe1428869343f57dea30b7da5b29d3'),
          (SD + 'train-student/student2-final-step012000.pth', 'd2c3c515b2249d7b85591fda602f24476a917c8fd6f4b89486aed613601973b0'),
          (SD + 'train-scramble/scramble2-final-step012000.pth', '7320ca4f891e6b205e60532b36188179b90cdc7d00484f25e335edf3773dcb76'),
          ('student2-test-ink9-maps.py', 'bfcb478fe17fcd0a272dc6b1880087225c84162faaaabd18e4a99676983d0bbe'),
          ('student-test-score.py', 'f798e0d01b0037cbc3aae06c6c71f97cec0e63ef06d9b1e2de5b8f168174cc48'),
          (R + '252405-stack.npy', '7b044a4df16d98b543fb18ba11b255a8e65ec96fb8438471d5c2c7c96210bdbb'),
          (T1 + 'arm-A-R1-tauil_ft6k-rev-mean5.npy', '5312246bcd3e364480c7832817e9f7c9727e93f44b2d91533af0cb1c54c34337'),
          (V1 + 'test/student-rev-mean5.npy', '110823285bb403dbb425709964a2136ae5ece2e0fe959473ecaa1f35d8ebde7f'),
          (V1 + 'test/scramble-rev-mean5.npy', '00e9b55255f711fc2c3195f1a21e90686e3fe103d9afa103b592919a9ed410f5'),
          (T1 + 'baseline-s43_75k-rev-mean5.npy', '16766c21eab8efbb0d86939506b21a9d0e261c9b7f67fdf2013ace844c7e090c')]
PX, E = 2.403 * 4, 40
SIGMAS = (48.0, 192.0)
SHIFTS = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
TILE, NBOOT = 208, 2000
Q_GATE, Q_V1 = 1.25, 2.5  # G1/G2 tail on the second look; v1's tail, reported for continuity
B_WANT = {'auc_sup': 0.8102, 'hp192_b2': 0.3499}
V1_WANT = {'auc_sup': 0.8252, 'hp192_b2': 0.4543}
SH21_48_FLOOR, SH41_48_FLOOR = 0.0293, 0.0207
SLOOK = 'second look at 252405'
S, Z, FWD, SH21, SH41 = 'student2 rev mean5', 'scramble2 rev mean5', 'student2 fwd mean5', 'student2_shuf21 rev mean5', 'student2_shuf41 rev mean5'
B, S43 = 'TAUIL ft6k rev mean5 (day-1 map; B for G2 G4 G5)', 'released s43_75k rev mean5 (day-1 map)'
BV1, ZV1 = 'STUDENT v1 rev mean5 (v1 export; B for G1)', 'scramble twin v1 rev mean5 (v1 export)'
TSP, RND = 'TEACHER T: ScrollPrize 2.4 um map (the SP key)', 'REF direct render, no model (layers 18-22)'

say('student2-test-score.py  %s  (%s)' % (time.strftime('%Y-%m-%d %H:%M:%S'), SLOOK))
for p, want in FROZEN:
    got = sha256(p)
    say('  %-62s sha256 %s (%s)' % (p, got, 'matches' if got == want else 'MISMATCH'))
    if got != want:
        stop('%s differs from its frozen sha256' % p)
st_arr = np.load(R + '252405-stack.npy', mmap_mode='r')
sha1 = hashlib.sha1(np.ascontiguousarray(st_arr).tobytes()).hexdigest()[:12]
say('  252405 stack array sha1 %s (render/252405-grid.json records 80e0f696ae14: %s)' % (sha1, 'equal' if sha1 == '80e0f696ae14' else 'DIFFERENT'))
if sha1 != '80e0f696ae14':
    stop('stack array sha1 differs')

# ---- twin null, from both training logs ------------------------------------------------------------
twin = {}
for arm in ('train-student', 'train-scramble'):
    txt = open(SD + arm + '/train-run.txt', encoding='utf-8').read()
    rs = [float(x) for x in re.findall(r'twin null check: hp r 192 um \(sigma 19\.975 px\) teacher vs rolled teacher over the training px ([+-]\d\.\d+)', txt)]
    ok = 'twin a valid null on every segment: True' in txt and 'COMPLETE' in txt.splitlines()[0]
    twin[arm] = {'r_w00_552896': rs, 'valid_line': ok}
twin_valid = all(v['valid_line'] and len(v['r_w00_552896']) == 2 and all(abs(r) <= 0.10 for r in v['r_w00_552896']) for v in twin.values())
say('  twin null (training logs): %s -> %s' % ('; '.join('%s r %s' % (k, ', '.join('%+.4f' % r for r in v['r_w00_552896'])) for k, v in twin.items()),
                                            'VALID (|r| <= 0.10 on both segments, both logs)' if twin_valid else 'NOT VALID'))

# ---- pixels and labels (day 1) --------------------------------------------------------------------
mask = np.load(R + '252405-mask.npy')
H, W = mask.shape
VAL = mask
M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
L = T1 + 'labels-hf/'
lab = np.load(L + '252405_inklabels_v2-grid.npy'); sup = np.load(L + '252405_supervision_mask_v2-grid.npy')
lab3 = np.load(R + '252405-inklabels-grid.npy'); sup3 = np.load(R + '252405-supervision-grid.npy')
pos = M & (lab >= 0.75); negS = M & (lab <= 0) & (sup >= 0.75)
pos3 = M & (lab3 >= 0.75); neg3 = M & (lab3 <= 0) & (sup3 >= 0.75)
sp = np.load(R + '252405-spmap-grid.npy').astype(np.float32) * 255
lj = json.load(open(L + 'labels-hf.json'))['ceiling']


def auc(s, p, n):
    v = np.concatenate([s[p], s[n]]).astype(np.float64)
    rk = rankdata(v)
    n1, n0 = int(p.sum()), int(n.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


C = auc(sp, pos, negS)
C3 = auc(sp, pos3, neg3)
say('  pixels %d (render mask %d, 40 px edge cut); HF v2 pos %d neg(sup) %d; S3 pos %d neg(sup) %d' % (M.sum(), mask.sum(), pos.sum(), negS.sum(), pos3.sum(), neg3.sum()))
say('  C = %.6f (labels part %.6f -> %s); S3 beside %.6f' % (C, lj['C'], 'equal' if abs(C - lj['C']) < 1e-9 else 'DIFFERENT', C3))
if not (abs(C - lj['C']) < 1e-9 and int(pos.sum()) == lj['pos'] and int(negS.sum()) == lj['neg_sup'] and int(M.sum()) == 4468325):
    stop('pixels, labels or C differ from day 1')

m32 = M.astype(np.float32)
Wv = {b: gaussian_filter(VAL.astype(np.float32), b) for b in (2, 4)}
PRE = {}
for sig_um in SIGMAS:
    sig = sig_um / PX
    G = np.maximum(gaussian_filter(m32, sig), 1e-3)
    core = M & (gaussian_filter(m32, 2 * sig) > 0.999)
    PRE[sig_um] = (sig, G, core, sp - gaussian_filter(sp * m32, sig) / G)
    say('  hp sigma %g um = %.3f px: core %d px' % (sig_um, sig, core.sum()))
# label key (G2)
sigL = 192.0 / PX
mL = M & ((sup >= 0.75) | (lab >= 0.75))
mL32 = mL.astype(np.float32)
GL = np.maximum(gaussian_filter(mL32, sigL), 1e-3)
coreL = mL & (gaussian_filter(mL32, 2 * sigL) > 0.999)
labf = lab.astype(np.float32)
khL = labf - gaussian_filter(labf * mL32, sigL) / GL
say('  label key K (HF v2 inklabels): m_L %d px, core_L %d px at 192 um' % (mL.sum(), coreL.sum()))


def mblur(x, b):
    if not b:
        return x
    return gaussian_filter(np.where(VAL, x, 0).astype(np.float32), b) / np.maximum(Wv[b], 1e-3)


def pr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])


def shifted(a, b, m, dy, dx):
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = m[ys:ye, xs:xe] & m[ys - dy:ye - dy, xs - dx:xe - dx]
    return float(np.corrcoef(a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm])[0, 1])


def summ(r0, nl):
    nl = np.asarray(nl)
    return {'r': r0, 'null_mean': float(nl.mean()), 'null_sd': float(nl.std()), 'null_min': float(nl.min()), 'null_max': float(nl.max()),
            'z': float((r0 - nl.mean()) / max(nl.std(), 1e-12)), 'n_null_ge': int((nl >= r0).sum()), 'above_max': bool(r0 > nl.max())}


def nulls(xh, kh, core):
    r0 = pr(xh, kh, core)
    wrap = [pr(xh, np.roll(kh, s, (0, 1)), core & np.roll(core, s, (0, 1))) for s in SHIFTS]
    over = [shifted(xh, kh, core, dy, dx) for dy, dx in SHIFTS]
    return {'wrap': summ(r0, wrap), 'overlap': summ(r0, over)}


HPX = {}


def hp_read(x, sig_um, b):
    sig, G, core, kh = PRE[sig_um]
    xb = mblur(np.asarray(x, np.float32), b)
    return xb - gaussian_filter(xb * m32, sig) / G


def hp_read_L(x):
    xb = mblur(np.asarray(x, np.float32), 2)
    return xb - gaussian_filter(xb * mL32, sigL) / GL


# ---- maps ----------------------------------------------------------------------------------------
found = {}
for d in MAPDIRS:
    for f in sorted(glob.glob(os.path.join(d, '*.npy'))):
        mm = re.match(r'(.+)-c([+-]\d+)-(rev|fwd)\.npy$', os.path.basename(f))
        if mm:
            found.setdefault((mm.group(1), mm.group(3)), {})[int(mm.group(2))] = f
maps, srcs, centre = {}, {}, {}
for (mod, o), d in sorted(found.items()):
    k = '%s %s mean5' % (mod, o)
    if all(c in d for c in (-4, -2, 0, 2, 4)):
        cm = [np.load(d[c]).astype(np.float32) for c in (-4, -2, 0, 2, 4)]
        maps[k] = np.mean(cm, 0)
        srcs[k] = {('c%+d' % c): {'path': d[c], 'sha256': sha256(d[c])} for c in (-4, -2, 0, 2, 4)}
        centre[k] = cm
    else:
        say('  INCOMPLETE %s: centres %s' % (k, sorted(d)))
for k in (S, Z, FWD, SH21, SH41):
    say('  %-28s %s' % (k, 'present' if k in maps else 'ABSENT'))
    if k not in maps:
        stop('map %s absent' % k)
maps[B] = np.load(T1 + 'arm-A-R1-tauil_ft6k-rev-mean5.npy').astype(np.float32)
maps[BV1] = np.load(V1 + 'test/student-rev-mean5.npy').astype(np.float32)
maps[ZV1] = np.load(V1 + 'test/scramble-rev-mean5.npy').astype(np.float32)
maps[S43] = np.load(T1 + 'baseline-s43_75k-rev-mean5.npy').astype(np.float32)
maps[TSP] = sp
maps[RND] = np.asarray(st_arr[18:23], np.float32).mean(0)
DRY = bool(np.array_equal(maps[S], maps[BV1]))
if DRY:
    say('  DRY RUN: S (%s) is byte-equal to the v1 student map, so S = B for G1 and every G1 number below is MEANINGLESS' % S)

ORDER = [S, Z, BV1, B, TSP, FWD, SH21, SH41, ZV1, S43, RND]
BLURS = {S: (0, 2, 4), Z: (0, 2, 4), BV1: (0, 2, 4), B: (0, 2, 4), TSP: (0, 2, 4), ZV1: (0, 2, 4)}
rows = {}
for name in ORDER:
    a = maps[name]
    row = {'map': name, 'auc_sup': auc(a, pos, negS), 'auc_all': auc(a, pos, M & (lab <= 0)), 'auc_sup_s3': auc(a, pos3, neg3)}
    row['nAUC'] = (row['auc_sup'] - 0.5) / (C - 0.5)
    row['nAUC_s3'] = (row['auc_sup_s3'] - 0.5) / (C3 - 0.5)
    for s in SIGMAS:
        for b in BLURS.get(name, (0, 2)):
            xh = hp_read(a, s, b)
            row['hp%d_b%d' % (s, b)] = nulls(xh, PRE[s][3], PRE[s][2])
            if b == 2:
                HPX[(name, s)] = xh
            row['hp%d_b%d' % (s, b)]['core_read_le0'] = int((PRE[s][2] & (a <= 0)).sum())
    if name in srcs:
        row['sources'] = srcs[name]
    if name in centre:
        row['per_centre_auc_sup'] = {('c%+d' % c): auc(cm, pos, negS) for c, cm in zip((-4, -2, 0, 2, 4), centre[name])}
    rows[name] = row
    say('  scored %-50s AUCsup %.4f nAUC %.4f  hp192 b2 %+.4f  hp48 b2 %+.4f' % (name, row['auc_sup'], row['nAUC'], row['hp192_b2']['wrap']['r'], row['hp48_b2']['wrap']['r']))

# ---- baseline reproduction -----------------------------------------------------------------------
bA, bH = round(rows[B]['auc_sup'], 4), round(rows[B]['hp192_b2']['wrap']['r'], 4)
say('')
say('  TAUIL (B for G2 G4 G5) reproduction: AUCsup %.6f -> %.4f (want %.4f); hp192 b2 %+.6f -> %+.4f (want %+.4f) -> %s' % (
    rows[B]['auc_sup'], bA, B_WANT['auc_sup'], rows[B]['hp192_b2']['wrap']['r'], bH, B_WANT['hp192_b2'],
    'REPRODUCED' if (bA == B_WANT['auc_sup'] and bH == B_WANT['hp192_b2']) else 'NOT REPRODUCED'))
if not (bA == B_WANT['auc_sup'] and bH == B_WANT['hp192_b2']):
    stop('the baseline does not reproduce at 4 dp')
vA, vH = round(rows[BV1]['auc_sup'], 4), round(rows[BV1]['hp192_b2']['wrap']['r'], 4)
say('  STUDENT v1 (B for G1) reproduction: AUCsup %.6f -> %.4f (want %.4f); hp192 b2 %+.6f -> %+.4f (want %+.4f) -> %s' % (
    rows[BV1]['auc_sup'], vA, V1_WANT['auc_sup'], rows[BV1]['hp192_b2']['wrap']['r'], vH, V1_WANT['hp192_b2'],
    'REPRODUCED' if (vA == V1_WANT['auc_sup'] and vH == V1_WANT['hp192_b2']) else 'NOT REPRODUCED'))
if not (vA == V1_WANT['auc_sup'] and vH == V1_WANT['hp192_b2']):
    stop('the v1 student map does not reproduce at 4 dp')

# ---- label-key hp (G2) ---------------------------------------------------------------------------
HPL = {}
for name in ORDER[:-1]:
    xh = hp_read_L(maps[name])
    HPL[name] = xh
    rows[name]['hpK192_b2'] = nulls(xh, khL, coreL)
say('  label-key hp192 b2: ' + '; '.join('%s %+.4f' % (n.split(':')[0], rows[n]['hpK192_b2']['wrap']['r']) for n in ORDER[:-1]))

# ---- tile bootstrap ------------------------------------------------------------------------------
nty, ntx = math.ceil(H / TILE), math.ceil(W / TILE)
NT = nty * ntx
tid = (np.arange(H)[:, None] // TILE) * ntx + (np.arange(W)[None, :] // TILE)


def draws(pool):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pool), size=(NBOOT, len(pool)))
    K = np.zeros((NBOOT, NT))
    for r in range(NBOOT):
        K[r] = np.bincount(pool[idx[r]], minlength=NT)
    return K


tp, tn = tid[pos], tid[negS]
Pc, Nc = np.bincount(tp, minlength=NT).astype(np.float64), np.bincount(tn, minlength=NT).astype(np.float64)


def auc_matrix(a):
    sp_ = a[pos].astype(np.float64)
    sn = a[negS].astype(np.float64)
    A = np.zeros((NT, NT))
    for u in np.unique(tn):
        nu = np.sort(sn[tn == u])
        lt = np.searchsorted(nu, sp_, 'left')
        le = np.searchsorted(nu, sp_, 'right')
        A[:, u] = np.bincount(tp, weights=lt + 0.5 * (le - lt), minlength=NT)
    return A


def boot_auc(A, K):
    return np.einsum('rt,tu,ru->r', K, A, K) / ((K @ Pc) * (K @ Nc))


def tile_sums(xh, kh, core):
    x = xh[core].astype(np.float64); y = kh[core].astype(np.float64); t = tid[core]
    x -= x.mean(); y -= y.mean()
    return np.stack([np.bincount(t, weights=w, minlength=NT) for w in (np.ones_like(x), x, y, x * x, y * y, x * y)])


def boot_r(Sm, K):
    n, sx, sy, sxx, syy, sxy = (K @ Sm[i] for i in range(6))
    return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))


def pct(c, q):
    s = np.sort(c)
    p = q / 100 * (len(s) - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    if not (np.isfinite(s[lo]) and np.isfinite(s[hi])):
        return -math.inf
    return float(s[lo] + (p - lo) * (s[hi] - s[lo]))


def closure_block(label, sv, bv, tv, s_pt, b_pt, t_pt, q=Q_GATE):
    d = tv - bv
    cl = np.where(d > 0, (sv - bv) / np.where(d > 0, d, 1), -np.inf)
    pt = (s_pt - b_pt) / (t_pt - b_pt) if t_pt - b_pt > 0 else -math.inf
    out = {'point_S': s_pt, 'point_B': b_pt, 'point_T': t_pt, 'point_T_minus_B': t_pt - b_pt, 'point_closure': pt,
           'boot_closure_p1_25': pct(cl, 1.25), 'boot_closure_p2_5': pct(cl, 2.5), 'boot_closure_p50': pct(cl, 50), 'boot_closure_p97_5': pct(cl, 97.5),
           'boot_n_neg_inf': int(np.isinf(cl).sum()), 'boot_S_minus_B_p2_5': pct(sv - bv, 2.5), 'boot_S_minus_B_p97_5': pct(sv - bv, 97.5),
           'boot_T_minus_B_p2_5': pct(d, 2.5), 'boot_T_minus_B_p97_5': pct(d, 97.5), 'boot_S_sd': float(np.std(sv)), 'boot_B_sd': float(np.std(bv)),
           'gate_tail_pct': q}
    out['boot_closure_gate_tail'] = pct(cl, q)
    out['pass'] = bool(pt >= 0.5 and out['boot_closure_gate_tail'] >= 0.25)
    say('  %s: S %.4f B %.4f T %.4f (T - B %.4f); closure point %.4f; bootstrap 1.25 / 2.5 / 50 / 97.5 pct %.4f / %.4f / %.4f / %.4f (%d of %d at -inf);'
        ' S - B 95%% [%.4f, %.4f] -> %s on the %g pct rule' % (label, s_pt, b_pt, t_pt, t_pt - b_pt, pt, out['boot_closure_p1_25'], out['boot_closure_p2_5'],
                                                             out['boot_closure_p50'], out['boot_closure_p97_5'], out['boot_n_neg_inf'], NBOOT,
                                                             out['boot_S_minus_B_p2_5'], out['boot_S_minus_B_p97_5'], 'PASS' if out['pass'] else 'FAIL', q))
    return out


say('')
say('  TILE BOOTSTRAP: %d x %d px tiles (2 mm at 9.612 um = %.3f mm), %d x %d = %d tiles anchored at (0, 0); %d resamples, default_rng(0) per pool'
    % (TILE, TILE, TILE * PX / 1000, nty, ntx, NT, NBOOT))
boot = {}
G1TAG = ' [DRY RUN: S = B, MEANINGLESS]' if DRY else ''
poolA = np.nonzero((Pc + Nc) > 0)[0]
KA = draws(poolA)
AM = {n: auc_matrix(maps[n]) for n in (S, BV1, B, TSP, Z, ZV1)}
for n, A in AM.items():
    chk = A.sum() / (Pc.sum() * Nc.sum())
    if abs(chk - rows[n]['auc_sup']) > 1e-9:
        stop('tile AUC matrix of %s gives %.12f, midrank %.12f' % (n, chk, rows[n]['auc_sup']))
say('  AUC pool: %d tiles hold a pos or a neg; tile AUC matrices reproduce the midrank AUCsup at k = 1 (|diff| <= 1e-9) for S, student v1, TAUIL, T, twin v2, twin v1' % len(poolA))
bv = {n: boot_auc(A, KA) for n, A in AM.items()}
boot['G1_auc'] = closure_block('G1 AUCsup closure (student v2, B = student v1)' + G1TAG, bv[S], bv[BV1], bv[TSP], rows[S]['auc_sup'], rows[BV1]['auc_sup'], rows[TSP]['auc_sup'], Q_GATE)
boot['auc_vs_tauil_v1rule_reported'] = closure_block('   AUCsup closure of student v2 against TAUIL, v1\'s 2.5th pct rule (REPORTED, never deciding)', bv[S], bv[B], bv[TSP],
                                                     rows[S]['auc_sup'], rows[B]['auc_sup'], rows[TSP]['auc_sup'], Q_V1)
boot['v1_student_auc_vs_tauil_reported'] = closure_block('   AUCsup closure of student v1 against TAUIL, v1\'s G1 recomputed (reported)', bv[BV1], bv[B], bv[TSP],
                                                         rows[BV1]['auc_sup'], rows[B]['auc_sup'], rows[TSP]['auc_sup'], Q_V1)
boot['twin_auc_reported'] = closure_block('   AUCsup closure of the twin v2 against TAUIL (reported)', bv[Z], bv[B], bv[TSP], rows[Z]['auc_sup'], rows[B]['auc_sup'], rows[TSP]['auc_sup'], Q_V1)

poolL = np.nonzero(np.bincount(tid[coreL], minlength=NT) > 0)[0]
KL = draws(poolL)
SL = {n: tile_sums(HPL[n], khL, coreL) for n in (S, BV1, B, TSP, Z)}
for n, Sm in SL.items():
    chk = boot_r(Sm, np.ones((1, NT)))[0]
    if abs(chk - rows[n]['hpK192_b2']['wrap']['r']) > 1e-9:
        stop('tile Pearson of %s on the label key gives %.12f, direct %.12f' % (n, chk, rows[n]['hpK192_b2']['wrap']['r']))
bL = {n: boot_r(Sm, KL) for n, Sm in SL.items()}
TK, BK = rows[TSP]['hpK192_b2']['wrap']['r'], rows[B]['hpK192_b2']['wrap']['r']
sdB, sdBo = rows[B]['hpK192_b2']['wrap']['null_sd'], rows[B]['hpK192_b2']['overlap']['null_sd']
informative = (TK - BK) >= 2 * sdB
say('  label-key pool: %d tiles hold a core_L px; tile Pearson reproduces the direct r at k = 1 for S, student v1, TAUIL, T, twin v2' % len(poolL))
say('  G2 informativeness (item 8, B = TAUIL): T_K %+.4f - B_K %+.4f = %.4f vs 2 x sd of B\'s 48 wrap nulls on core_L = 2 x %.4f = %.4f -> %s'
    '  (overlap null sd %.4f -> 2 x = %.4f, %s)' % (TK, BK, TK - BK, sdB, 2 * sdB, 'INFORMATIVE' if informative else 'UNINFORMATIVE',
                                                     sdBo, 2 * sdBo, 'informative' if (TK - BK) >= 2 * sdBo else 'uninformative'))
boot['G2_hpK'] = closure_block('G2 hp192 b2 closure, label key (student v2, B = TAUIL)', bL[S], bL[B], bL[TSP], rows[S]['hpK192_b2']['wrap']['r'], BK, TK, Q_GATE)
boot['G2_hpK']['informative'] = bool(informative)
boot['hpK_vs_v1_reported'] = closure_block('   label-key closure of student v2 against student v1 (reported)', bL[S], bL[BV1], bL[TSP],
                                           rows[S]['hpK192_b2']['wrap']['r'], rows[BV1]['hpK192_b2']['wrap']['r'], TK, Q_V1)
boot['twin_hpK_reported'] = closure_block('   label-key closure of the twin v2 against TAUIL (reported)', bL[Z], bL[B], bL[TSP], rows[Z]['hpK192_b2']['wrap']['r'], BK, TK, Q_V1)

c192 = PRE[192.0][2]
poolS = np.nonzero(np.bincount(tid[c192], minlength=NT) > 0)[0]
KS = draws(poolS)
SS = {n: tile_sums(HPX[(n, 192.0)], PRE[192.0][3], c192) for n in (S, BV1, B, TSP, Z)}
for n, Sm in SS.items():
    chk = boot_r(Sm, np.ones((1, NT)))[0]
    if abs(chk - rows[n]['hp192_b2']['wrap']['r']) > 1e-9:
        stop('tile Pearson of %s on the SP key gives %.12f, direct %.12f' % (n, chk, rows[n]['hp192_b2']['wrap']['r']))
bS = {n: boot_r(Sm, KS) for n, Sm in SS.items()}
say('  SP-key pool: %d tiles hold a 192 um core px' % len(poolS))
boot['SPkey_literal_reported'] = closure_block('   SP-key literal hp192 b2 closure of student v2 against TAUIL (REPORTED, never deciding)', bS[S], bS[B], bS[TSP],
                                               rows[S]['hp192_b2']['wrap']['r'], rows[B]['hp192_b2']['wrap']['r'], rows[TSP]['hp192_b2']['wrap']['r'], Q_V1)
boot['SPkey_literal_vs_v1_reported'] = closure_block('   SP-key literal hp192 b2 closure of student v2 against student v1 (REPORTED, never deciding)', bS[S], bS[BV1], bS[TSP],
                                                     rows[S]['hp192_b2']['wrap']['r'], rows[BV1]['hp192_b2']['wrap']['r'], rows[TSP]['hp192_b2']['wrap']['r'], Q_V1)

# ---- gates ---------------------------------------------------------------------------------------
g = {}
g['G1'] = boot['G1_auc']['pass']
g['G2'] = 'UNINFORMATIVE' if not informative else ('PASS' if boot['G2_hpK']['pass'] else 'FAIL')
h = rows[S]['hp192_b2']
g['G3'] = bool(h['wrap']['above_max'] and h['overlap']['above_max'])
g4 = {}
undefined_r = []


def inside(r, bw, bo, name):
    # prereg-v2 G4_undefined_r: an r with no variance behind it reads no structure and counts inside, named
    if not (np.isfinite(r['wrap']['r']) and np.isfinite(r['overlap']['r'])):
        undefined_r.append(name)
        return True
    return bool(r['wrap']['r'] <= bw and r['overlap']['r'] <= bo)


for k in (FWD, SH21, SH41):
    r = rows[k]['hp192_b2']
    g4['%s 192 b2' % k] = inside(r, r['wrap']['null_max'], r['overlap']['null_max'], '%s 192 b2' % k)
for k, floor in ((FWD, None), (SH21, SH21_48_FLOOR), (SH41, SH41_48_FLOOR)):
    r = rows[k]['hp48_b2']
    bw = r['wrap']['null_max'] if floor is None else max(r['wrap']['null_max'], floor)
    bo = r['overlap']['null_max'] if floor is None else max(r['overlap']['null_max'], floor)
    g4['%s 48 b2' % k] = inside(r, bw, bo, '%s 48 b2' % k)
    g4['%s 48 b2 bars' % k] = (bw, bo)
g['G4'] = all(v for kk, v in g4.items() if not kk.endswith('bars'))
g['G4_undefined_r'] = undefined_r
g5_bar = rows[B]['nAUC'] + 0.02
g['G5'] = bool(rows[Z]['nAUC'] <= g5_bar)
g['twin_valid_null'] = twin_valid


def rd(x):
    return '%+.4f' % x if np.isfinite(x) else 'undefined (no variance)'


say('')
say('  GATES (%s)' % SLOOK)
say('    G1 AUCsup closure, B = student v1, >= 0.5 and 1.25th pct >= 0.25: %s%s (point %.4f, 1.25th %.4f, 2.5th %.4f; S %.4f vs the point bar %.6f)' % (
    'PASS' if g['G1'] else 'FAIL', G1TAG, boot['G1_auc']['point_closure'], boot['G1_auc']['boot_closure_p1_25'], boot['G1_auc']['boot_closure_p2_5'],
    rows[S]['auc_sup'], rows[BV1]['auc_sup'] + 0.5 * (rows[TSP]['auc_sup'] - rows[BV1]['auc_sup'])))
say('    G2 label-key hp192 b2 closure, B = TAUIL: %s (point %.4f, 1.25th %.4f; S_K %+.4f)' % (g['G2'], boot['G2_hpK']['point_closure'], boot['G2_hpK']['boot_closure_p1_25'],
                                                                                       rows[S]['hpK192_b2']['wrap']['r']))
say('    G3 student v2 hp192 b2 (SP key) %+.4f above null max wrap %+.4f AND overlap %+.4f: %s' % (h['wrap']['r'], h['wrap']['null_max'], h['overlap']['null_max'], 'PASS' if g['G3'] else 'FAIL'))
for k in (FWD, SH21, SH41):
    r1, r4 = rows[k]['hp192_b2'], rows[k]['hp48_b2']
    bw, bo = g4['%s 48 b2 bars' % k]
    say('    G4 %-26s 192 b2 %s <= [wrap %+.4f, overlap %+.4f]: %s;  48 b2 %s <= [wrap %+.4f, overlap %+.4f]: %s' % (
        k, rd(r1['wrap']['r']), r1['wrap']['null_max'], r1['overlap']['null_max'], 'inside' if g4['%s 192 b2' % k] else 'OUTSIDE',
        rd(r4['wrap']['r']), bw, bo, 'inside' if g4['%s 48 b2' % k] else 'OUTSIDE'))
say('    G4 controls: %s%s' % ('PASS' if g['G4'] else 'FAIL', ('; undefined r counted inside: ' + ', '.join(undefined_r)) if undefined_r else '; no undefined r'))
say('    G5 twin v2 nAUC %.4f <= TAUIL nAUC %.4f + 0.02 = %.4f (AUCsup bar %.4f): %s' % (rows[Z]['nAUC'], rows[B]['nAUC'], g5_bar, 0.5 + g5_bar * (C - 0.5), 'PASS' if g['G5'] else 'FAIL'))
say('    twin v2 a valid null: %s' % ('YES' if twin_valid else 'NO'))
say('    G6 letter read: NOT RUN HERE (a separate run, fresh readers; pool from the exported maps)')

if not twin_valid:
    verdict = 'INCONCLUSIVE, %s (the twin is not a valid null)' % SLOOK
elif not (g['G1'] and g['G2'] in ('PASS', 'UNINFORMATIVE') and g['G3'] and g['G4'] and g['G5']):
    failed = [n for n, ok in (('G1', g['G1']), ('G2', g['G2'] in ('PASS', 'UNINFORMATIVE')), ('G3', g['G3']), ('G4', g['G4']), ('G5', g['G5'])) if not ok]
    verdict = 'NO-GO, %s (numeric gate%s %s failed; the letter read is still run and reported, and cannot turn this into GO)' % (SLOOK, 's' if len(failed) > 1 else '', ', '.join(failed))
else:
    verdict = 'NUMERIC GATES PASS, %s (G1-G5, twin valid%s): the verdict is GO or NUMBERS GO, LETTER NO-GO, decided by G6 (the letter read, not yet run)' % (
        SLOOK, ', G2 uninformative so GO rests on G1 plus the read' if g['G2'] == 'UNINFORMATIVE' else '')
if DRY:
    verdict = 'DRY RUN on staged v1 maps, MEANINGLESS (S = B for G1): ' + verdict
say('')
say('  VERDICT (numeric): %s' % verdict)

# ---- report tables -------------------------------------------------------------------------------
say('')
say('%-52s %7s %7s %7s %7s %7s | %-32s | %-32s | %s' % ('map', 'AUCsup', 'nAUC', 'AUCall', 'AUCs3', 'nAUCs3', 'hp192 b2 SP key r [wrap|ovl max] z',
                                                       'hp48 b2 SP key r [wrap|ovl max] z', 'hp192 b2 label key r [wrap max]'))
for name in ORDER:
    r = rows[name]
    a, b4 = r['hp192_b2'], r['hp48_b2']
    k = r.get('hpK192_b2')
    say('%-52s %7.4f %7.4f %7.4f %7.4f %7.4f | %+.4f [%+.4f|%+.4f] %5.1f | %+.4f [%+.4f|%+.4f] %5.1f | %s' % (
        name, r['auc_sup'], r['nAUC'], r['auc_all'], r['auc_sup_s3'], r['nAUC_s3'], a['wrap']['r'], a['wrap']['null_max'], a['overlap']['null_max'], a['wrap']['z'],
        b4['wrap']['r'], b4['wrap']['null_max'], b4['overlap']['null_max'], b4['wrap']['z'],
        ('%+.4f [%+.4f] sd %.4f' % (k['wrap']['r'], k['wrap']['null_max'], k['wrap']['null_sd'])) if k else '-'))
say('')
say('  blur 0 and 4 beside (hp r [wrap null max]):')
for name in ORDER:
    r = rows[name]
    say('    %-52s hp192 b0 %+.4f [%+.4f]  hp48 b0 %+.4f [%+.4f]%s' % (
        name, r['hp192_b0']['wrap']['r'], r['hp192_b0']['wrap']['null_max'], r['hp48_b0']['wrap']['r'], r['hp48_b0']['wrap']['null_max'],
        ('  hp192 b4 %+.4f [%+.4f]  hp48 b4 %+.4f [%+.4f]' % (r['hp192_b4']['wrap']['r'], r['hp192_b4']['wrap']['null_max'],
                                                             r['hp48_b4']['wrap']['r'], r['hp48_b4']['wrap']['null_max'])) if 'hp192_b4' in r else ''))
say('  core px with read <= 0 (192 / 48 um, b2): ' + '; '.join('%s %d/%d' % (n.split(':')[0][:20], rows[n]['hp192_b2']['core_read_le0'], rows[n]['hp48_b2']['core_read_le0']) for n in ORDER))
say('')
say('  per-centre AUCsup (HF v2):')
for name in (S, Z, FWD, SH21, SH41):
    say('    %-28s %s' % (name, '  '.join('%s %.4f' % (c, v) for c, v in rows[name]['per_centre_auc_sup'].items())))

# ---- exports for the letter read -----------------------------------------------------------------
from PIL import Image
exp = {'written': time.strftime('%Y-%m-%dT%H:%M:%S'), 'grid': [H, W], 'pitch_um': 9.612, 'mask': R + '252405-mask.npy', 'files': {}}
for stem, name in (('student2-rev-mean5', S), ('scramble2-rev-mean5', Z), ('depth-null-student2-shuf21-rev-mean5', SH21)):
    p = os.path.join(EXPORT_DIR, stem + '.npy')
    np.save(p, maps[name])
    q = os.path.join(EXPORT_DIR, stem + '-raw8.png')
    Image.fromarray(np.clip(np.rint(maps[name]), 0, 255).astype(np.uint8)).save(q)
    exp['files'][stem] = {'npy': p, 'npy_sha256': sha256(p), 'sources': {os.path.basename(v['path']): v['sha256'] for v in srcs[name].values()},
                          'mean_in_mask': float(maps[name][mask].mean()), 'sd_in_mask': float(maps[name][mask].std()), 'png': q, 'png_sha256': sha256(q)}
    say('  exported %s (sha256 %s) and %s (sha256 %s)' % (p, exp['files'][stem]['npy_sha256'], q, exp['files'][stem]['png_sha256']))
exp['note'] = ('float32 maps on infer.py\'s 0-255 scale, five-centre rev mean, unmasked, 2125 x 2265 at 9.612 um; the -raw8.png is the map '
               'rounded and clipped to uint8, nothing else (as day 1). depth-null-student2-shuf21 is the prereg-v2 G6 depth null for make-pool.py --depth-map. ' + SLOOK + '.')
json.dump(exp, open(os.path.join(EXPORT_DIR, 'exports.json'), 'w'), indent=1)

json.dump({'second_look': SLOOK, 'dry_run': DRY, 'frozen': FROZEN, 'C': C, 'C_s3': C3, 'n_pixels': int(M.sum()), 'pos': int(pos.sum()), 'neg_sup': int(negS.sum()),
           'n_core': {str(int(s)): int(PRE[s][2].sum()) for s in SIGMAS}, 'label_key': {'m_L': int(mL.sum()), 'core_L': int(coreL.sum())},
           'twin': twin, 'twin_valid': twin_valid, 'boot': boot, 'tiles': {'size': TILE, 'grid': [nty, ntx], 'pool_auc': len(poolA), 'pool_hpK': len(poolL),
                                                                         'pool_hpSP': len(poolS), 'resamples': NBOOT},
           'gates': {k: v for k, v in g.items()}, 'g4_detail': {k: v for k, v in g4.items()}, 'g5_bar_nAUC': g5_bar, 'verdict_numeric': verdict,
           'rows': rows}, open(os.path.splitext(OUT)[0] + '.json', 'w'), indent=1, default=float)
runfile.done(OUT, '\n'.join(lines) + '\n')
