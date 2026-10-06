"""day1-test-score.py - prereg v2's decisive scoring of the frozen arms on 252405, with the controls and the
released s43-75k baseline on the same pixels, and its decision line applied arm by arm.

WHAT IS SCORED. Every <name>-c<+d>-<order>.npy in the map dirs given (day1-ink9-maps.py output); per
(name, order) the five-centre mean (-4,-2,0,+2,+4), the per-member map of prereg v2 and of the frozen
recipe. Arm A (selected recipe R1) and arm B (best single member) are the same map, tauil_ft6k rev mean5
(test/recipe-frozen.json, identical_to_A), so one row serves both and the verdict is printed under each label.

PIXELS AND LABELS (the labels part's, rebuilt here and checked against its JSON): render mask minus a 40 px
edge; HF v2 grids from test/labels-hf; pos = label >= 0.75; neg = label <= 0 AND supervision >= 0.75;
midrank AUC. C = AUCsup of ScrollPrize's 2.4 um map (render/252405-spmap-grid.npy) on the same pixels, recomputed
here and required to equal labels-hf.json's 0.86895. nAUC = (AUCsup - 0.5) / (C - 0.5). S3 20260918 grids beside.

hp r: day1-score-baselines.py's, the scorer every selection row used. hp(x) = x - G(x m)/G(m) on m = the scored
pixels; core = m where G(m, 2 sigma) > 0.999; read blurred (masked Gaussian on the render mask) by 0 or 2 px
before the high-pass, key never blurred. Key = render/252405-spmap-grid.npy. Nulls, 48 shifts of 104 px steps
to 312 px per axis: WRAP (np.roll of key and core, day1-score-baselines.py, prereg v2's "rolled-key") and
OVERLAP (shifted crop, no wrap, gates/prereg.json's literal "overlap only (no wrap)"). Both are computed.

DECISION LINE (prereg v2 decisive_on_252405, arm by arm), with the readings this script takes where it is silent:
  GO         nAUC >= 0.771 AND hp r 192 um blur 2 > null max AND forward-order and depth-shuffled controls inside
  AMBIGUOUS  0.661 <= nAUC < 0.771; and nAUC >= 0.771 with a failed hp or control condition (gates/prereg.json's
             fill-in, which v2 does not contradict)
  EXHAUSTED  nAUC < 0.661
  [reading] "above the null max": above BOTH the wrap and the overlap null max (v2 says rolled-key, gates/prereg.json
            says overlap; the stricter of the two is taken).
  [reading] "control inside its null": gates/prereg.json's pass rule, hp r at blur 2 <= its own null max, taken at
            192 um (v2's decisive band) AND 48 um, against BOTH nulls. The two-sided form is printed beside.
  [reading] the depth-shuffled control is gates/prereg.json's (each 21-layer slab permuted by default_rng(0).
            permutation(21) before the rev flip); the task's 41-layer permutation (seed 0) also gates.
  Sanity gates 1 and 2 (prereg v2 sanity_gates_before_scoring) are read from their own run files; the prereg is
  silent on a failure, so the decision line is applied as written and a GO beside a failed gate is ALSO reported as
  "ambiguous (gate failed)", the conservative reading.

Usage: day1-test-score.py <out.txt> <maps dir> [<maps dir> ...]     CPU only, no model. Run from <workdir>.
Writes <out.txt> and <out>.json.
"""
import glob
import hashlib
import json
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


PREREG, PREREG_SHA = 'geom/day1-9um/prereg-v2.json', '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'
RECIPE, RECIPE_SHA = 'geom/day1-9um/test/recipe-frozen.json', '72ad196adb48ce3e356b435f3b087c2b92b95b252186f7e9bbd960f74724e897'
R, T = 'geom/day1-9um/render/', 'geom/day1-9um/test/'
GO_N, AMB_N = 0.771, 0.661
PX, E = 2.403 * 4, 40
SIGMAS = (48.0, 192.0)
BLURS = (0, 2)
SHIFTS = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
ARM = 'tauil_ft6k rev mean5'
BASE = 's43_75k rev mean5'
CTRL_FWD = 'tauil_ft6k fwd mean5'
CTRL_SH21 = 'tauil_ft6k_shuf21 rev mean5'
CTRL_SH41 = 'tauil_ft6k_shuf41 rev mean5'

say('day1-test-score.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
for p, want in ((PREREG, PREREG_SHA), (RECIPE, RECIPE_SHA)):
    got = sha256(p)
    say('  %s sha256 %s (%s)' % (p, got, 'matches' if got == want else 'MISMATCH'))
    if got != want:
        runfile.done(OUT, '\n'.join(lines) + '\nSTOPPED: frozen file differs.\n')
        sys.exit(2)
rec = json.load(open(RECIPE))
assert rec['arms']['B_best_single_member']['identical_to_A'] and rec['arms']['A_selected_recipe']['members'][0]['name'] == 'tauil_ft6k'

mask = np.load(R + '252405-mask.npy')
H, W = mask.shape
VAL = mask
M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
L = T + 'labels-hf/'
lab = np.load(L + '252405_inklabels_v2-grid.npy'); sup = np.load(L + '252405_supervision_mask_v2-grid.npy')
val = np.load(L + '252405_validation_mask_v2-grid.npy')
lab3 = np.load(R + '252405-inklabels-grid.npy'); sup3 = np.load(R + '252405-supervision-grid.npy')
pos = M & (lab >= 0.75); negS = M & (lab <= 0) & (sup >= 0.75)
pos3 = M & (lab3 >= 0.75); neg3 = M & (lab3 <= 0) & (sup3 >= 0.75)
sp = np.load(R + '252405-spmap-grid.npy').astype(np.float32) * 255
lj = json.load(open(T + 'labels-hf/labels-hf.json'))['ceiling']


def auc(s, p, n):
    v = np.concatenate([s[p], s[n]]).astype(np.float64)
    rk = rankdata(v)
    n1, n0 = int(p.sum()), int(n.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


C = auc(sp, pos, negS)
C3 = auc(sp, pos3, neg3)
say('  pixels %d (render mask %d, 40 px edge cut); HF v2 pos %d neg(sup) %d; S3 pos %d neg(sup) %d; validation >= 0.75 on %d of them'
    % (M.sum(), mask.sum(), pos.sum(), negS.sum(), pos3.sum(), neg3.sum(), (M & (val >= 0.75)).sum()))
say('  C (SP 2.4 um map, HF v2) = %.6f; labels part %.6f -> %s.  S3 beside: %.6f (labels part %.6f)'
    % (C, lj['C'], 'equal' if abs(C - lj['C']) < 1e-9 else 'DIFFERENT', C3, lj['C_s3']))
assert abs(C - lj['C']) < 1e-9 and int(pos.sum()) == lj['pos'] and int(negS.sum()) == lj['neg_sup'], 'pixels/labels differ from the labels part'
say('  bars: nAUC %.3f -> raw AUCsup %.4f (go); nAUC %.3f -> %.4f (ambiguous floor)' % (GO_N, 0.5 + GO_N * (C - 0.5), AMB_N, 0.5 + AMB_N * (C - 0.5)))

m32 = M.astype(np.float32)
Wv = {b: gaussian_filter(VAL.astype(np.float32), b) for b in BLURS if b}
PRE = {}
for sig_um in SIGMAS:
    sig = sig_um / PX
    G = np.maximum(gaussian_filter(m32, sig), 1e-3)
    core = M & (gaussian_filter(m32, 2 * sig) > 0.999)
    kh = sp - gaussian_filter(sp * m32, sig) / G
    PRE[sig_um] = (sig, G, core, kh)
    say('  hp sigma %g um = %.2f px: core %d px' % (sig_um, sig, core.sum()))


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
            'z': float((r0 - nl.mean()) / max(nl.std(), 1e-12)), 'n_null_ge': int((nl >= r0).sum()),
            'above_max': bool(r0 > nl.max()), 'inside_one_sided': bool(r0 <= nl.max()), 'inside_two_sided': bool(nl.min() <= r0 <= nl.max())}


def hpr(x, sig_um, b):
    sig, G, core, kh = PRE[sig_um]
    xb = mblur(np.asarray(x, np.float32), b)
    xh = xb - gaussian_filter(xb * m32, sig) / G
    r0 = pr(xh, kh, core)
    wrap = [pr(xh, np.roll(kh, s, (0, 1)), core & np.roll(core, s, (0, 1))) for s in SHIFTS]
    over = [shifted(xh, kh, core, dy, dx) for dy, dx in SHIFTS]
    return {'wrap': summ(r0, wrap), 'overlap': summ(r0, over), 'core_read_le0': int((core & (x <= 0)).sum())}


# ---- maps -----------------------------------------------------------------------------------------
found = {}
for d in MAPDIRS:
    for f in sorted(glob.glob(os.path.join(d, '*.npy'))):
        mm = re.match(r'(.+)-c([+-]\d+)-(rev|fwd)\.npy$', os.path.basename(f))
        if mm:
            found.setdefault((mm.group(1), mm.group(3)), {})[int(mm.group(2))] = f
maps, srcs = {}, {}
for (mod, o), d in sorted(found.items()):
    if all(c in d for c in (-4, -2, 0, 2, 4)):
        cm = [np.load(d[c]).astype(np.float32) for c in (-4, -2, 0, 2, 4)]
        maps['%s %s mean5' % (mod, o)] = np.mean(cm, 0)
        srcs['%s %s mean5' % (mod, o)] = {('c%+d' % c): {'path': d[c], 'sha256': sha256(d[c])} for c in (-4, -2, 0, 2, 4)}
        for c, a in zip((-4, -2, 0, 2, 4), cm):
            maps['%s %s c%+d' % (mod, o, c)] = a
    else:
        say('  INCOMPLETE %s %s: centres %s' % (mod, o, sorted(d)))
for k in (ARM, BASE, CTRL_FWD, CTRL_SH21, CTRL_SH41):
    say('  %-30s %s' % (k, 'present' if k in maps else 'ABSENT'))

# gate-2 map vs this run's s43_75k c+0 rev (same recipe, separate GPU run)
g2p = T + 'gate2/252405-ink9-s43-75k-c+0-rev.npy'
g2_vs = None
if os.path.exists(g2p) and 's43_75k rev c+0' in maps:
    g2 = np.load(g2p)
    dd = np.abs(g2 - maps['s43_75k rev c+0'])
    g2_vs = {'max_abs_diff': float(dd.max()), 'mean_abs_diff': float(dd.mean()), 'r': float(np.corrcoef(g2[M], maps['s43_75k rev c+0'][M])[0, 1])}
    say('  gate-2 s43-75k c+0 map vs this run\'s s43_75k c+0 rev: max |diff| %.4g, mean |diff| %.4g, r %.6f' % (g2_vs['max_abs_diff'], g2_vs['mean_abs_diff'], g2_vs['r']))

stack = np.load(R + '252405-stack.npy', mmap_mode='r')
refs = {'REF ScrollPrize 2.4 um map (the key; AUCsup = C)': sp, 'REF direct render, no model (layers 18-22)': np.asarray(stack[18:23], np.float32).mean(0)}

order = [k for k in (ARM, BASE, CTRL_FWD, CTRL_SH21, CTRL_SH41) if k in maps] + list(refs)
rows = {}
say('')
say('%-46s %8s %7s %8s %7s | %-34s | %-34s' % ('map', 'AUCsup', 'nAUC', 'AUCs3', 'nAUCs3', 'hp192 b2: r [wrap max | overlap max]', 'hp48 b2: r [wrap max | overlap max]'))
for name in order:
    a = maps.get(name, refs.get(name))
    row = {'map': name, 'auc_sup': auc(a, pos, negS), 'auc_all': auc(a, pos, M & (lab <= 0)), 'auc_sup_s3': auc(a, pos3, neg3)}
    row['nAUC'] = (row['auc_sup'] - 0.5) / (C - 0.5)
    row['nAUC_s3'] = (row['auc_sup_s3'] - 0.5) / (C3 - 0.5)
    for s in SIGMAS:
        for b in BLURS:
            row['hp%d_b%d' % (s, b)] = hpr(a, s, b)
    if name in srcs:
        row['sources'] = srcs[name]
    rows[name] = row
    h1, h4 = row['hp192_b2'], row['hp48_b2']
    say('%-46s %8.4f %7.4f %8.4f %7.4f | %+.4f [%+.4f | %+.4f] z %4.1f | %+.4f [%+.4f | %+.4f] z %4.1f' % (
        name, row['auc_sup'], row['nAUC'], row['auc_sup_s3'], row['nAUC_s3'], h1['wrap']['r'], h1['wrap']['null_max'], h1['overlap']['null_max'],
        h1['wrap']['z'], h4['wrap']['r'], h4['wrap']['null_max'], h4['overlap']['null_max'], h4['wrap']['z']))
say('')
say('  blur 0 beside (hp r [wrap null max]):')
for name in order:
    row = rows[name]
    say('    %-46s hp192 b0 %+.4f [%+.4f]   hp48 b0 %+.4f [%+.4f]   core px with read <= 0: %d (192) %d (48)' % (
        name, row['hp192_b0']['wrap']['r'], row['hp192_b0']['wrap']['null_max'], row['hp48_b0']['wrap']['r'], row['hp48_b0']['wrap']['null_max'],
        row['hp192_b2']['core_read_le0'], row['hp48_b2']['core_read_le0']))
say('')
say('  per-centre AUCsup (HF v2), beside:')
for mod in ('tauil_ft6k rev', 's43_75k rev', 'tauil_ft6k fwd', 'tauil_ft6k_shuf21 rev', 'tauil_ft6k_shuf41 rev'):
    cs = ['%s c%+d' % (mod, c) for c in (-4, -2, 0, 2, 4)]
    if all(k in maps for k in cs):
        say('    %-24s %s' % (mod, '  '.join('c%+d %.4f' % (c, auc(maps[k], pos, negS)) for c, k in zip((-4, -2, 0, 2, 4), cs))))

# ---- sanity gates ---------------------------------------------------------------------------------
g1 = json.load(open(T + 'gate1.json'))
g2j = T + 'gate2/252405-gate2.json'
g2 = json.load(open(g2j)) if os.path.exists(g2j) else None
gate1_pass = bool(g1['pass_conservative'])
gate2_pass = bool(g2['pass']) if g2 else None
say('')
say('  SANITY GATES: gate 1 %s (test/gate1.json, conservative; narrow day-1 rule %s); gate 2 %s%s' % (
    'PASS' if gate1_pass else 'FAIL', 'PASS' if g1['pass_narrow_day1_rule'] else 'FAIL',
    {True: 'PASS', False: 'FAIL', None: 'NOT RUN'}[gate2_pass],
    (' (peak r %+.4f at %s px, r(0,0) %+.4f)' % (g2['peak_r'], tuple(g2['peak_shift_px']), g2['r_zero'])) if g2 else ''))


# ---- decision ------------------------------------------------------------------------------------
def ctrl_check(name):
    if name not in rows:
        return None
    r = rows[name]
    letter = {'%s %s' % (k, n): r[k][n]['inside_one_sided'] for k in ('hp192_b2', 'hp48_b2') for n in ('wrap', 'overlap')}
    two = {'%s %s' % (k, n): r[k][n]['inside_two_sided'] for k in ('hp192_b2', 'hp48_b2') for n in ('wrap', 'overlap')}
    return {'inside': all(letter.values()), 'inside_192_only': r['hp192_b2']['wrap']['inside_one_sided'] and r['hp192_b2']['overlap']['inside_one_sided'],
            'inside_192_wrap_only': r['hp192_b2']['wrap']['inside_one_sided'],
            'inside_two_sided_all': all(two.values()), 'detail_one_sided': letter, 'detail_two_sided': two}


def classify(n, hp_ok, ctrl_ok):
    if n >= GO_N and hp_ok and ctrl_ok:
        return 'GO'
    if n >= GO_N:
        return 'AMBIGUOUS'
    if n >= AMB_N:
        return 'AMBIGUOUS'
    return 'EXHAUSTED'


ctrls = {k: ctrl_check(k) for k in (CTRL_FWD, CTRL_SH21, CTRL_SH41)}
say('')
say('  CONTROLS (arm tauil_ft6k; inside = hp r at blur 2 <= its own null max, 192 AND 48 um, wrap AND overlap):')
for k, v in ctrls.items():
    if v is None:
        say('    %-30s NOT RUN' % k)
        continue
    r = rows[k]
    say('    %-30s %s   (192 only: %s; two-sided at both: %s)  AUCsup %.4f nAUC %.4f; hp192 b2 %+.4f [wrap %+.4f..%+.4f, overlap max %+.4f]; hp48 b2 %+.4f [wrap max %+.4f, overlap max %+.4f]' % (
        k, 'INSIDE' if v['inside'] else 'OUTSIDE', 'inside' if v['inside_192_only'] else 'OUTSIDE', 'inside' if v['inside_two_sided_all'] else 'OUTSIDE',
        r['auc_sup'], r['nAUC'], r['hp192_b2']['wrap']['r'], r['hp192_b2']['wrap']['null_min'], r['hp192_b2']['wrap']['null_max'], r['hp192_b2']['overlap']['null_max'],
        r['hp48_b2']['wrap']['r'], r['hp48_b2']['wrap']['null_max'], r['hp48_b2']['overlap']['null_max']))
ctrl_prereg = [ctrls[CTRL_FWD], ctrls[CTRL_SH21]]
ctrl_ok_prereg = all(c is not None and c['inside'] for c in ctrl_prereg)
ctrl_ok_all = ctrl_ok_prereg and ctrls[CTRL_SH41] is not None and ctrls[CTRL_SH41]['inside']
ctrl_ok_narrow = all(c is not None and c['inside_192_wrap_only'] for c in ctrl_prereg)

decisions = {}
say('')
say('  DECISION LINE (prereg v2), arm by arm:')
for label, name in (('arm A (selected recipe R1 = tauil_ft6k)', ARM), ('arm B (best single member = tauil_ft6k)', ARM), ('baseline released s43-75k (reported, not an arm)', BASE)):
    if name not in rows:
        say('    %s: MAP ABSENT' % label)
        continue
    r = rows[name]
    h = r['hp192_b2']
    hp_ok = h['wrap']['above_max'] and h['overlap']['above_max']
    hp_ok_wrap = h['wrap']['above_max']
    if name == ARM:
        v = classify(r['nAUC'], hp_ok, ctrl_ok_all)
        v_wrap192 = classify(r['nAUC'], hp_ok_wrap, ctrl_ok_narrow)
    else:
        v = classify(r['nAUC'], hp_ok, True) if r['nAUC'] < GO_N else ('GO-LEVEL nAUC; controls not run for the baseline' if hp_ok else 'AMBIGUOUS')
        v_wrap192 = None
    gated = v
    if v == 'GO' and (not gate1_pass or gate2_pass is not True):
        gated = 'AMBIGUOUS (decision line GO, but sanity gate %s failed; conservative reading)' % (
            ' and '.join(g for g, ok in (('1', gate1_pass), ('2', gate2_pass is True)) if not ok))
    decisions[label] = {'map': name, 'auc_sup': r['auc_sup'], 'nAUC': r['nAUC'], 'auc_sup_s3': r['auc_sup_s3'], 'nAUC_s3': r['nAUC_s3'],
                        'nAUC_ge_go': r['nAUC'] >= GO_N, 'nAUC_ge_ambiguous': r['nAUC'] >= AMB_N,
                        'hp192_b2_r': h['wrap']['r'], 'hp192_b2_wrap_null_max': h['wrap']['null_max'], 'hp192_b2_overlap_null_max': h['overlap']['null_max'],
                        'hp_above_null_max_both': hp_ok, 'hp_above_wrap_null_max': hp_ok_wrap,
                        'controls_inside_prereg_fwd_shuf21': ctrl_ok_prereg if name == ARM else None,
                        'controls_inside_all_incl_shuf41': ctrl_ok_all if name == ARM else None,
                        'verdict_decision_line': v, 'verdict_narrow_reading_wrap_192_only': v_wrap192, 'verdict_with_sanity_gates': gated}
    say('    %s:' % label)
    say('      AUCsup %.4f  nAUC %.4f  (go >= %.3f: %s; ambiguous >= %.3f: %s)   S3 beside: AUCsup %.4f nAUC %.4f' % (
        r['auc_sup'], r['nAUC'], GO_N, 'yes' if r['nAUC'] >= GO_N else 'no', AMB_N, 'yes' if r['nAUC'] >= AMB_N else 'no', r['auc_sup_s3'], r['nAUC_s3']))
    say('      hp r 192 um b2 %+.4f vs null max wrap %+.4f / overlap %+.4f -> %s' % (h['wrap']['r'], h['wrap']['null_max'], h['overlap']['null_max'],
                                                                                   'ABOVE both' if hp_ok else ('above wrap only' if hp_ok_wrap else 'NOT above')))
    if name == ARM:
        say('      controls: forward %s, depth-shuffled (21-layer, prereg) %s, depth-shuffled (41-layer, task) %s' % tuple(
            ('inside' if c['inside'] else 'OUTSIDE') if c else 'not run' for c in (ctrls[CTRL_FWD], ctrls[CTRL_SH21], ctrls[CTRL_SH41])))
    say('      VERDICT (decision line): %s' % v)
    if v_wrap192 is not None and v_wrap192 != v:
        say('      (narrowest reading, wrap null only, 192 um only, prereg controls only: %s)' % v_wrap192)
    say('      VERDICT with the sanity gates: %s' % gated)

json.dump({'prereg_sha256': PREREG_SHA, 'recipe_sha256': RECIPE_SHA, 'C': C, 'C_s3': C3, 'bars_nAUC': {'go': GO_N, 'ambiguous': AMB_N},
           'bars_raw': {'go': 0.5 + GO_N * (C - 0.5), 'ambiguous': 0.5 + AMB_N * (C - 0.5)}, 'n_pixels': int(M.sum()), 'pos': int(pos.sum()),
           'neg_sup': int(negS.sum()), 'pos_s3': int(pos3.sum()), 'neg_sup_s3': int(neg3.sum()),
           'n_core': {str(int(s)): int(PRE[s][2].sum()) for s in SIGMAS}, 'gate1_pass_conservative': gate1_pass, 'gate2': g2,
           'gate2_map_vs_run': g2_vs, 'controls': ctrls, 'decisions': decisions, 'rows': rows},
          open(os.path.splitext(OUT)[0] + '.json', 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
