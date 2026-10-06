"""student-sceptic-rescore.py - the sceptic's independent re-score of the student branch's decisive 252405 numbers,
from the SAVED maps only (no model, no GPU). prereg-student.json (f3ebf0ab...d3710) + amendment 1 (3b0f1bc4...cf10).

Own code, not the scorer's: Gaussian = separable FFT convolution in float64 with zero padding (truncate 4 sigma, the
same kernel definition), AUC = tie-grouped Mann-Whitney over unique values, Pearson in float64, nulls by index
gather (wrap = modular partner, overlap = in-range partner), bootstrap AUC = pixel-weighted over the resampled tiles.
Definitions (pixels, labels, hp, core, shifts, gates) are the prereg's, read from section 5 and 6.

Usage: student-sceptic-rescore.py      CPU only. Writes geom/student-9um/sceptic/rescore.txt and rescore.json.
"""
import hashlib
import json
import math
import os
import time

import numpy as np
from scipy.signal import fftconvolve

ROOT = './'
SD = ROOT + 'geom/student-9um/'
RD = ROOT + 'geom/day1-9um/render/'
T1 = ROOT + 'geom/day1-9um/test/'
OUTD = SD + 'sceptic/'
os.makedirs(OUTD, exist_ok=True)
lines = []
J = {}


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


t0 = time.time()
say('student-sceptic-rescore.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
FROZEN = [(SD + 'prereg-student.json', 'f3ebf0abc3f93a6772ce31f6915d7f5aef2ea4afc72badeb11cbb735216d3710'),
          (SD + 'prereg-student-amendment-1.txt', '3b0f1bc434decb39e9b46e87ef5913c4bd53eb9c1a649a3ccb93f6d69f82cf10'),
          (SD + 'test/student-rev-mean5.npy', '110823285bb403dbb425709964a2136ae5ece2e0fe959473ecaa1f35d8ebde7f'),
          (SD + 'test/scramble-rev-mean5.npy', '00e9b55255f711fc2c3195f1a21e90686e3fe103d9afa103b592919a9ed410f5'),
          (SD + 'test/depth-null-student-shuf21-rev-mean5.npy', '42b2a3860fa1c486b307738355eb0ab74b5cc4815c2ca580edf6c3603e1ee207'),
          (T1 + 'arm-A-R1-tauil_ft6k-rev-mean5.npy', '5312246bcd3e364480c7832817e9f7c9727e93f44b2d91533af0cb1c54c34337')]
for p, want in FROZEN:
    got = sha(p)
    say('  %-70s %s' % (p.replace(ROOT, ''), 'sha256 matches' if got == want else 'MISMATCH %s' % got))
    assert got == want, p

# ---- the centre maps behind each mean5, hashed against exports.json -----------------------------------
exp = json.load(open(SD + 'test/exports.json'))
src_ok = True
for stem, d in exp['files'].items():
    for fn, h in d['sources'].items():
        sub = 'maps-shuf21/' if 'shuf21' in fn else 'maps/'
        g = sha(SD + 'test/' + sub + fn)
        src_ok &= (g == h)
say('  centre maps behind the three exports hash equal to exports.json: %s' % src_ok)
assert src_ok


def load_mean5(sub, name, order):
    cs = [np.load(SD + 'test/%s/%s-c%+d-%s.npy' % (sub, name, c, order)) for c in (-4, -2, 0, 2, 4)]
    m64 = np.mean(np.stack([c.astype(np.float64) for c in cs]), 0)
    return m64, cs


mask = np.load(RD + '252405-mask.npy')
H, W = mask.shape
E = 40
M = mask.copy()
M[:E] = False; M[-E:] = False; M[:, :E] = False; M[:, -E:] = False
lab = np.load(T1 + 'labels-hf/252405_inklabels_v2-grid.npy').astype(np.float64)
sup = np.load(T1 + 'labels-hf/252405_supervision_mask_v2-grid.npy').astype(np.float64)
pos = M & (lab >= 0.75)
neg = M & (lab <= 0) & (sup >= 0.75)
sp = np.load(RD + '252405-spmap-grid.npy').astype(np.float32).astype(np.float64) * 255.0
say('  pixels %d  pos %d  neg(sup) %d   (prereg: 4,468,325 / 279,497 / 940,946)' % (M.sum(), pos.sum(), neg.sum()))
assert (M.sum(), pos.sum(), neg.sum()) == (4468325, 279497, 940946)

maps = {}
mS64, cS = load_mean5('maps', 'student', 'rev')
maps['S'] = np.load(SD + 'test/student-rev-mean5.npy').astype(np.float64)
say('  student export vs my float64 mean of its 5 centre maps: max |diff| %.3g' % np.abs(maps['S'] - mS64).max())
mZ64, _ = load_mean5('maps', 'scramble', 'rev')
maps['Z'] = np.load(SD + 'test/scramble-rev-mean5.npy').astype(np.float64)
say('  twin export vs my mean: max |diff| %.3g' % np.abs(maps['Z'] - mZ64).max())
maps['B'] = np.load(T1 + 'arm-A-R1-tauil_ft6k-rev-mean5.npy').astype(np.float64)
maps['T'] = sp
maps['FWD'], _ = load_mean5('maps', 'student', 'fwd')
maps['SH21'], _ = load_mean5('maps-shuf21', 'student_shuf21', 'rev')
ex21 = np.load(SD + 'test/depth-null-student-shuf21-rev-mean5.npy').astype(np.float64)
say('  shuf21 export vs my mean: max |diff| %.3g' % np.abs(ex21 - maps['SH21']).max())
maps['SH41'], _ = load_mean5('maps-shuf41', 'student_shuf41', 'rev')
NAMES = {'S': 'student rev mean5', 'Z': 'twin rev mean5', 'B': 'baseline B (day-1 frozen)', 'T': 'teacher T (SP 2.4 um)',
         'FWD': 'student fwd mean5', 'SH21': 'student shuf21 rev mean5', 'SH41': 'student shuf41 rev mean5'}

# ---- the controls' inputs: permutations as the prereg defines them ----------------------------------
p21 = np.random.default_rng(0).permutation(21).tolist()
p41 = np.random.default_rng(0).permutation(41).tolist()
log21 = open(SD + 'test/maps-shuf21/maps-log.txt').read()
ok21 = str(p21) in log21
s41 = json.load(open(SD + 'test/shuffle41/shuf41.json'))
st = np.load(RD + '252405-stack.npy', mmap_mode='r')
st41 = np.load(SD + 'test/shuffle41/252405-stack-shuf41.npy', mmap_mode='r')
ok41 = s41['perm'] == p41 and all(np.array_equal(st41[k], st[p41[k]]) for k in range(41))
say('  shuffle-21 order in its map log = default_rng(0).permutation(21): %s;  shuffle-41 stack[k] == stack[perm[k]] for all 41, perm = default_rng(0).permutation(41): %s' % (ok21, ok41))
J['controls_inputs_ok'] = {'shuf21': ok21, 'shuf41': bool(ok41)}


# ---- my AUC: tie-grouped Mann-Whitney ------------------------------------------------------------------
class AUCer:
    def __init__(self, a):
        vp, vn = a[pos], a[neg]
        u, inv = np.unique(np.concatenate([vp, vn]), return_inverse=True)
        self.n = len(u)
        self.ip, self.ineg = inv[:len(vp)], inv[len(vp):]

    def __call__(self, wp=None, wn=None):
        Wp = np.bincount(self.ip, weights=wp, minlength=self.n)
        Wn = np.bincount(self.ineg, weights=wn, minlength=self.n)
        below = np.cumsum(Wn) - Wn
        return float(np.dot(Wp, below + 0.5 * Wn) / (Wp.sum() * Wn.sum()))


aucer = {k: AUCer(v) for k, v in maps.items()}
AUC = {k: f() for k, f in aucer.items()}
C = AUC['T']
nAUC = {k: (v - 0.5) / (C - 0.5) for k, v in AUC.items()}
say('')
say('  C (AUCsup of the SP map) = %.6f   (prereg 0.868948)' % C)
for k in maps:
    say('  %-28s AUCsup %.6f  nAUC %.6f' % (NAMES[k], AUC[k], nAUC[k]))


# ---- my Gaussian: separable FFT, float64, zero padding ------------------------------------------------
def gk(sig):
    r = int(4.0 * sig + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sig) ** 2)
    return w / w.sum()


def gb(a, sig):
    k = gk(sig)
    t = fftconvolve(np.asarray(a, np.float64), k[:, None], mode='same')
    return fftconvolve(t, k[None, :], mode='same')


PX = 2.403 * 4
Vf = mask.astype(np.float64)
Mf = M.astype(np.float64)
VB = {b: np.maximum(gb(Vf, b), 1e-3) for b in (2,)}


def readb(x, b):
    return gb(np.where(mask, x, 0.0), b) / VB[b]


PRE = {}
for sig_um in (192.0, 48.0):
    s = sig_um / PX
    Gm = np.maximum(gb(Mf, s), 1e-3)
    core = M & (gb(Mf, 2 * s) > 0.999)
    PRE[sig_um] = (s, Gm, core, sp - gb(sp * Mf, s) / Gm)
    say('  hp %g um (sigma %.4f px): core %d px' % (sig_um, s, core.sum()))
sL = 192.0 / PX
mL = M & ((sup >= 0.75) | (lab >= 0.75))
mLf = mL.astype(np.float64)
GL = np.maximum(gb(mLf, sL), 1e-3)
coreL = mL & (gb(mLf, 2 * sL) > 0.999)
khL = lab - gb(lab * mLf, sL) / GL
say('  label key: m_L %d px, core_L %d px   (scorer: 1,226,005 / 503,062)' % (mL.sum(), coreL.sum()))

SHIFTS = [(dy, dx) for dy in range(-312, 313, 104) for dx in range(-312, 313, 104) if (dy, dx) != (0, 0)]
assert len(SHIFTS) == 48


def pear(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))


def rnull(xh, kh, core):
    yy, xx = np.nonzero(core)
    xv = xh[yy, xx]
    r0 = pear(xv, kh[yy, xx])
    wrap, over = [], []
    for dy, dx in SHIFTS:
        qy, qx = yy - dy, xx - dx
        inr = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
        qy %= H
        qx %= W
        ok = core[qy, qx]
        wrap.append(pear(xv[ok], kh[qy[ok], qx[ok]]))
        o2 = ok & inr
        over.append(pear(xv[o2], kh[qy[o2], qx[o2]]))
    wrap, over = np.array(wrap), np.array(over)
    return {'r': r0, 'wrap_max': float(wrap.max()), 'over_max': float(over.max()), 'wrap_sd': float(wrap.std()),
            'over_sd': float(over.std()), 'wrap_mean': float(wrap.mean())}


HP = {}
HPX = {}
say('')
for k in ('S', 'Z', 'B', 'T', 'FWD', 'SH21', 'SH41'):
    xb = readb(maps[k], 2)
    for sig_um in (192.0, 48.0):
        s, Gm, core, kh = PRE[sig_um]
        xh = xb - gb(xb * Mf, s) / Gm
        HP[(k, sig_um)] = rnull(xh, kh, core)
        if sig_um == 192.0:
            HPX[k] = xh
    if k in ('S', 'Z', 'B', 'T'):
        xhL = xb - gb(xb * mLf, sL) / GL
        HP[(k, 'L')] = rnull(xhL, khL, coreL)
        HPX[(k, 'L')] = xhL
    h1, h4 = HP[(k, 192.0)], HP[(k, 48.0)]
    say('  %-28s hp192 b2 %+.4f [wrap max %+.4f | ovl max %+.4f]   hp48 b2 %+.4f [wrap max %+.4f | ovl max %+.4f]%s' % (
        NAMES[k], h1['r'], h1['wrap_max'], h1['over_max'], h4['r'], h4['wrap_max'], h4['over_max'],
        ('   label key hp192 b2 %+.4f [wrap max %+.4f, sd %.4f; ovl sd %.4f]' % (HP[(k, 'L')]['r'], HP[(k, 'L')]['wrap_max'], HP[(k, 'L')]['wrap_sd'], HP[(k, 'L')]['over_sd'])) if (k, 'L') in HP else ''))
say('  [%.0f s]' % (time.time() - t0))

# ---- baseline reproduction --------------------------------------------------------------------------
repro = round(AUC['B'], 4) == 0.8102 and round(HP[('B', 192.0)]['r'], 4) == 0.3499
say('')
say('  baseline reproduces AUCsup 0.8102 and hp192 b2 +0.3499 at 4 dp: %s (%.6f, %+.6f)' % (repro, AUC['B'], HP[('B', 192.0)]['r']))

# ---- tile bootstrap ---------------------------------------------------------------------------------
TILE, NB = 208, 2000
nty, ntx = -(-H // TILE), -(-W // TILE)
NT = nty * ntx
tid = (np.arange(H)[:, None] // TILE) * ntx + (np.arange(W)[None, :] // TILE)


def kmat(pool, scheme):
    n = len(pool)
    if scheme == 'replica':
        idx = np.random.default_rng(0).integers(0, n, size=(NB, n))
    else:
        g = np.random.default_rng(0)
        idx = np.stack([g.choice(n, size=n, replace=True) for _ in range(NB)])
    K = np.zeros((NB, NT))
    for r in range(NB):
        np.add.at(K[r], pool[idx[r]], 1.0)
    return K


def pctl(c, q):
    s = np.sort(c)
    p = q / 100 * (len(s) - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    if not np.isfinite(s[lo]):
        return -math.inf
    return float(s[lo] + (p - lo) * (s[hi] - s[lo]))


def closure(sv, bv, tv, sp_, bp_, tp_):
    d = tv - bv
    cl = np.full(len(d), -np.inf)
    good = d > 0
    cl[good] = (sv[good] - bv[good]) / d[good]
    pt = (sp_ - bp_) / (tp_ - bp_) if tp_ > bp_ else -math.inf
    return {'point': pt, 'p2_5': pctl(cl, 2.5), 'p50': pctl(cl, 50), 'p97_5': pctl(cl, 97.5), 'n_neg_inf': int((~good).sum()),
            'SmB_p2_5': pctl(sv - bv, 2.5), 'SmB_p97_5': pctl(sv - bv, 97.5)}


tpos, tneg = tid[pos], tid[neg]
poolA = np.nonzero(np.bincount(np.concatenate([tpos, tneg]), minlength=NT) > 0)[0]
poolL = np.nonzero(np.bincount(tid[coreL], minlength=NT) > 0)[0]
c192 = PRE[192.0][2]
poolS = np.nonzero(np.bincount(tid[c192], minlength=NT) > 0)[0]
say('')
say('  tiles %d x %d = %d; pools: AUC %d, label key %d, SP key %d   (scorer: 53, 26, 110)' % (nty, ntx, NT, len(poolA), len(poolL), len(poolS)))


def tile_stats(xh, kh, core):
    x = xh[core].astype(np.float64)
    y = kh[core].astype(np.float64)
    t = tid[core]
    x = x - x.mean()
    y = y - y.mean()
    out = np.zeros((6, NT))
    for i, w in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
        np.add.at(out[i], t, w)
    return out


def wr(St, K):
    n, sx, sy, sxx, syy, sxy = (K @ St[i] for i in range(6))
    return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))


BOOT = {}
for scheme in ('replica', 'alt-choice'):
    KA = kmat(poolA, scheme)
    bv = {}
    for k in ('S', 'B', 'T', 'Z'):
        f = aucer[k]
        vals = np.empty(NB)
        for r in range(NB):
            vals[r] = f(KA[r][tpos], KA[r][tneg])
        bv[k] = vals
    g1 = closure(bv['S'], bv['B'], bv['T'], AUC['S'], AUC['B'], AUC['T'])
    gz = closure(bv['Z'], bv['B'], bv['T'], AUC['Z'], AUC['B'], AUC['T'])
    KL = kmat(poolL, scheme)
    STL = {k: tile_stats(HPX[(k, 'L')], khL, coreL) for k in ('S', 'B', 'T')}
    bl = {k: wr(v, KL) for k, v in STL.items()}
    g2 = closure(bl['S'], bl['B'], bl['T'], HP[('S', 'L')]['r'], HP[('B', 'L')]['r'], HP[('T', 'L')]['r'])
    KS = kmat(poolS, scheme)
    STS = {k: tile_stats(HPX[k], PRE[192.0][3], c192) for k in ('S', 'B', 'T')}
    bs = {k: wr(v, KS) for k, v in STS.items()}
    gs = closure(bs['S'], bs['B'], bs['T'], HP[('S', 192.0)]['r'], HP[('B', 192.0)]['r'], HP[('T', 192.0)]['r'])
    BOOT[scheme] = {'G1_auc': g1, 'twin_auc': gz, 'G2_labelkey': g2, 'SPkey_literal': gs}
    for lab_, d in (('G1 AUCsup closure', g1), ('twin AUCsup closure', gz), ('G2 label-key closure', g2), ('SP-key literal closure', gs)):
        say('  [%s] %-24s point %.4f  2.5/50/97.5 pct %.4f / %.4f / %.4f  (%d at -inf)  S-B 95%% [%+.4f, %+.4f]' % (
            scheme, lab_, d['point'], d['p2_5'], d['p50'], d['p97_5'], d['n_neg_inf'], d['SmB_p2_5'], d['SmB_p97_5']))
    say('  [%.0f s]' % (time.time() - t0))

# ---- gates --------------------------------------------------------------------------------------------
g = {}
rb = BOOT['replica']
g['G1'] = rb['G1_auc']['point'] >= 0.5 and rb['G1_auc']['p2_5'] >= 0.25
TKmB = HP[('T', 'L')]['r'] - HP[('B', 'L')]['r']
inform = TKmB >= 2 * HP[('B', 'L')]['wrap_sd']
g['G2'] = 'UNINFORMATIVE' if not inform else ('PASS' if (rb['G2_labelkey']['point'] >= 0.5 and rb['G2_labelkey']['p2_5'] >= 0.25) else 'FAIL')
h = HP[('S', 192.0)]
g['G3'] = h['r'] > h['wrap_max'] and h['r'] > h['over_max']
g4 = {}
for k in ('FWD', 'SH21', 'SH41'):
    a = HP[(k, 192.0)]
    g4[k + ' 192'] = a['r'] <= a['wrap_max'] and a['r'] <= a['over_max']
for k, fl in (('FWD', None), ('SH21', 0.0293), ('SH41', 0.0207)):
    a = HP[(k, 48.0)]
    bw = a['wrap_max'] if fl is None else max(a['wrap_max'], fl)
    bo = a['over_max'] if fl is None else max(a['over_max'], fl)
    g4[k + ' 48'] = a['r'] <= bw and a['r'] <= bo
g['G4'] = all(g4.values())
g['G5'] = nAUC['Z'] <= nAUC['B'] + 0.02
tw = []
for arm in ('train-student', 'train-scramble'):
    txt = open(SD + arm + '/train-run.txt', encoding='utf-8').read()
    import re
    tw += [float(x) for x in re.findall(r'teacher vs rolled teacher over the training px ([+-]\d\.\d+)', txt)]
g['twin_valid'] = len(tw) == 4 and all(abs(x) <= 0.10 for x in tw)
say('')
say('  GATES (my numbers)')
say('    G1: closure point %.4f (bar 0.5), 2.5th pct %.4f (bar 0.25); point bar on S: %.6f -> %s' % (
    rb['G1_auc']['point'], rb['G1_auc']['p2_5'], AUC['B'] + 0.5 * (AUC['T'] - AUC['B']), 'PASS' if g['G1'] else 'FAIL'))
say('    G2: T_K - B_K = %+.4f vs 2 x sd(B wrap nulls) = %.4f -> %s' % (TKmB, 2 * HP[('B', 'L')]['wrap_sd'], g['G2']))
say('    G3: S hp192 b2 %+.4f vs null max %+.4f / %+.4f -> %s' % (h['r'], h['wrap_max'], h['over_max'], 'PASS' if g['G3'] else 'FAIL'))
for k, v in g4.items():
    say('    G4 %-9s %s' % (k, 'inside' if v else 'OUTSIDE'))
say('    G4 -> %s' % ('PASS' if g['G4'] else 'FAIL'))
say('    G5: twin nAUC %.4f vs B nAUC %.4f + 0.02 = %.4f -> %s' % (nAUC['Z'], nAUC['B'], nAUC['B'] + 0.02, 'PASS' if g['G5'] else 'FAIL'))
say('    twin valid null (training logs, 4 values %s): %s' % (tw, g['twin_valid']))
if not g['twin_valid']:
    verdict = 'INCONCLUSIVE'
elif not (g['G1'] and g['G2'] in ('PASS', 'UNINFORMATIVE') and g['G3'] and g['G4'] and g['G5']):
    verdict = 'NO-GO (failed: %s)' % ', '.join(n for n, ok in (('G1', g['G1']), ('G2', g['G2'] != 'FAIL'), ('G3', g['G3']), ('G4', g['G4']), ('G5', g['G5'])) if not ok)
else:
    verdict = 'numeric gates pass; G6 decides'
say('')
say('  VERDICT (numeric, my numbers): %s' % verdict)
say('  took %.0f s' % (time.time() - t0))

J.update({'C': C, 'AUC': AUC, 'nAUC': nAUC, 'hp': {'%s %s' % (k, s): v for (k, s), v in HP.items()}, 'boot': BOOT, 'gates': {k: (v if isinstance(v, str) else bool(v)) for k, v in g.items()},
          'g4': {k: bool(v) for k, v in g4.items()}, 'twin_logs': tw, 'baseline_reproduced': bool(repro), 'verdict': verdict,
          'pools': [len(poolA), len(poolL), len(poolS)], 'cores': {'192': int(PRE[192.0][2].sum()), '48': int(PRE[48.0][2].sum()), 'L': int(coreL.sum())}})
open(OUTD + 'rescore.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(J, open(OUTD + 'rescore.json', 'w'), indent=1, default=float)
