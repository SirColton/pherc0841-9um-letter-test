"""day1-sceptic-test.py - sceptic (numbers) re-check of PART test's decisive numbers on 252405.

Re-computes, with code written apart from day1-test-score.py / day1-gate1-full.py / day1-render-gate2.py
(own midrank AUC by value counts, own float64 Pearson, own shift loops), from the saved maps and grids:
  C; arm and baseline AUCsup / nAUC; hp r at 192 and 48 um, blur 2, with 48-shift wrap and overlap nulls;
  the forward and both depth-shuffle controls; gate 1 (direct render 18-22: raw r, hp 48 / 192 at blur 0);
  gate 2 (s43-75k c+0 rev vs the 2.4 um map, both Gaussian sigma 4, local shift scan); the shuffles'
  permutations; the decision line under each reading. Diagnostics, not rules: C and the arm's AUCsup
  against the HF grid with the map shifted by up to 4 px.
CPU only, no model. Usage: day1-sceptic-test.py <out.txt>   (run from <workdir>; writes <out>.json too)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

OUT = sys.argv[1]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
J = {}


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


R, T = 'geom/day1-9um/render/', 'geom/day1-9um/test/'
say('day1-sceptic-test.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
for p, want in (('geom/day1-9um/prereg-v2.json', '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'),
                (T + 'recipe-frozen.json', '72ad196adb48ce3e356b435f3b087c2b92b95b252186f7e9bbd960f74724e897')):
    g = sha256(p)
    say('  %s %s' % (p, 'matches' if g == want else 'MISMATCH ' + g))
    assert g == want

PX = 2.403 * 4
E = 40
STEPS = (-312, -208, -104, 0, 104, 208, 312)
SHIFTS = [(a, b) for a in STEPS for b in STEPS if (a, b) != (0, 0)]

mask = np.load(R + '252405-mask.npy').astype(bool)
H, W = mask.shape
M = np.zeros_like(mask)
M[E:H - E, E:W - E] = mask[E:H - E, E:W - E]
lab = np.load(T + 'labels-hf/252405_inklabels_v2-grid.npy')
sup = np.load(T + 'labels-hf/252405_supervision_mask_v2-grid.npy')
val = np.load(T + 'labels-hf/252405_validation_mask_v2-grid.npy')
lab3 = np.load(R + '252405-inklabels-grid.npy')
sup3 = np.load(R + '252405-supervision-grid.npy')
key = np.load(R + '252405-spmap-grid.npy').astype(np.float64)
POS = M & (lab >= 0.75)
NEG = M & (lab <= 0) & (sup >= 0.75)
POS3 = M & (lab3 >= 0.75)
NEG3 = M & (lab3 <= 0) & (sup3 >= 0.75)
say('  px %d; HF pos %d neg %d; S3 pos %d neg %d; validation>=0.75 on %d of M; key>0 on %d of M'
    % (M.sum(), POS.sum(), NEG.sum(), POS3.sum(), NEG3.sum(), (M & (val >= 0.75)).sum(), (M & (key > 0)).sum()))
J['px'] = int(M.sum()); J['pos'] = int(POS.sum()); J['neg'] = int(NEG.sum())


def auc(s, p, n):
    """Mann-Whitney U / (n1 n0) with half credit for ties, by value counts (no rankdata)."""
    a = np.asarray(s[p], np.float64)
    b = np.asarray(s[n], np.float64)
    u, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    cp = np.bincount(inv[:a.size], minlength=u.size).astype(np.float64)
    cn = np.bincount(inv[a.size:], minlength=u.size).astype(np.float64)
    below = np.concatenate([[0.0], np.cumsum(cn)[:-1]])
    return float((cp * (below + 0.5 * cn)).sum() / (a.size * b.size))


def pear(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


C = auc(key, POS, NEG)
C3 = auc(key, POS3, NEG3)
go_raw, amb_raw = 0.5 + 0.771 * (C - 0.5), 0.5 + 0.661 * (C - 0.5)
say('  C (HF v2) %.6f   C (S3) %.6f   raw bars: go %.4f  ambiguous %.4f' % (C, C3, go_raw, amb_raw))
J['C'] = C; J['C_s3'] = C3; J['bars_raw'] = [go_raw, amb_raw]

# maps from the centre files
cen = (-4, -2, 0, 2, 4)


def mean5(d, name, o):
    return np.mean([np.load('%s%s-c%+d-%s.npy' % (d, name, c, o)).astype(np.float32) for c in cen], 0)


maps = {'arm tauil rev': mean5(T + 'maps/', 'tauil_ft6k', 'rev'),
        'base s43_75k rev': mean5(T + 'maps/', 's43_75k', 'rev'),
        'ctl fwd': mean5(T + 'maps/', 'tauil_ft6k', 'fwd'),
        'ctl shuf21': mean5(T + 'maps-shuf21/', 'tauil_ft6k_shuf21', 'rev'),
        'ctl shuf41': mean5(T + 'maps-shuf41/', 'tauil_ft6k_shuf41', 'rev')}
armA = np.load(T + 'arm-A-R1-tauil_ft6k-rev-mean5.npy')
base_saved = np.load(T + 'baseline-s43_75k-rev-mean5.npy')
dshuf = np.load(T + 'depth-null-tauil_ft6k-shuf21-rev-mean5.npy')
say('  saved arm-A vs mean5 of centre files: max|d| %.3g; baseline %.3g; depth-null shuf21 %.3g'
    % (np.abs(armA - maps['arm tauil rev']).max(), np.abs(base_saved - maps['base s43_75k rev']).max(), np.abs(dshuf - maps['ctl shuf21']).max()))
say('  r(arm rev, shuf21) over M %.4f; r(arm rev, shuf41) %.4f; r(arm rev, fwd) %.4f'
    % (pear(maps['arm tauil rev'][M], maps['ctl shuf21'][M]), pear(maps['arm tauil rev'][M], maps['ctl shuf41'][M]),
       pear(maps['arm tauil rev'][M], maps['ctl fwd'][M])))
stack = np.load(R + '252405-stack.npy')
direct = stack[18:23].astype(np.float32).mean(0)

# permutations
p21 = np.random.default_rng(0).permutation(21).tolist()
p41 = np.random.default_rng(0).permutation(41)
s41j = json.load(open(T + 'shuffle41/shuf41.json'))
st41 = np.load(T + 'shuffle41/252405-stack-shuf41.npy', mmap_mode='r')
ok41 = all(np.array_equal(st41[k], stack[p41[k]]) for k in range(41))
say('  perm21 %s (log says [10, 18, 16, 4, 2, 12, 6, 19, 3, 11, 8, 0, 17, 13, 7, 20, 5, 14, 9, 1, 15])' % p21)
say('  perm41 == shuf41.json: %s; shuffled stack == stack[perm41] on all 41 layers: %s; identity layers in the c+-4 slabs: %s'
    % (p41.tolist() == s41j['perm'], ok41, sorted(set(k for k in range(6, 35) if p41[k] == k))))
J['perm21'] = p21; J['shuf41_ok'] = bool(ok41)

# AUC rows
say('')
say('%-18s %8s %8s %8s %8s' % ('map', 'AUCsup', 'nAUC', 'AUCs3', 'nAUCs3'))
rows = {}
for k, a in list(maps.items()) + [('ref direct 18-22', direct), ('ref key', key)]:
    r = {'auc': auc(a, POS, NEG), 'auc_s3': auc(a, POS3, NEG3)}
    r['nauc'] = (r['auc'] - 0.5) / (C - 0.5)
    r['nauc_s3'] = (r['auc_s3'] - 0.5) / (C3 - 0.5)
    rows[k] = r
    say('%-18s %8.4f %8.4f %8.4f %8.4f' % (k, r['auc'], r['nauc'], r['auc_s3'], r['nauc_s3']))
for k in ('arm tauil rev', 'base s43_75k rev'):
    say('  %s nAUC if C were 0.8609 (prereg text) %.4f; if 0.8787 (S3 48 um) %.4f'
        % (k, (rows[k]['auc'] - 0.5) / 0.3609, (rows[k]['auc'] - 0.5) / 0.3787))

# hp r
m = M.astype(np.float64)
VALf = mask.astype(np.float64)
pre = {}
for su in (48.0, 192.0):
    sg = su / PX
    Gm = np.maximum(gaussian_filter(m, sg), 1e-3)
    core = M & (gaussian_filter(m, 2 * sg) > 0.999)
    kh = key - gaussian_filter(key * m, sg) / Gm
    pre[su] = (sg, Gm, core, kh)
    say('  hp %g um: sigma %.3f px, core %d' % (su, sg, core.sum()))


def blur(x, b):
    if b == 0:
        return np.asarray(x, np.float64)
    return gaussian_filter(np.where(mask, x, 0).astype(np.float64), b) / np.maximum(gaussian_filter(VALf, b), 1e-3)


def crop_r(a, b, mm, dy, dx):
    """r between a[y, x] and b[y - dy, x - dx] where both positions are in mm."""
    ya, yb = max(dy, 0), H + min(dy, 0)
    xa, xb = max(dx, 0), W + min(dx, 0)
    A = a[ya:yb, xa:xb]; B = b[ya - dy:yb - dy, xa - dx:xb - dx]
    q = mm[ya:yb, xa:xb] & mm[ya - dy:yb - dy, xa - dx:xb - dx]
    return pear(A[q], B[q])


def nulls(xh, kh, core):
    r0 = pear(xh[core], kh[core])
    wr = np.array([pear(xh[core & np.roll(core, s, (0, 1))], np.roll(kh, s, (0, 1))[core & np.roll(core, s, (0, 1))]) for s in SHIFTS])
    ov = np.array([crop_r(xh, kh, core, dy, dx) for dy, dx in SHIFTS])
    return {'r': r0, 'wrap_max': float(wr.max()), 'wrap_min': float(wr.min()), 'wrap_sd': float(wr.std()), 'wrap_mean': float(wr.mean()),
            'over_max': float(ov.max()), 'over_min': float(ov.min()), 'n_ge_wrap': int((wr >= r0).sum())}


def hp(x, su, b):
    sg, Gm, core, kh = pre[su]
    xb = blur(x, b)
    xh = xb - gaussian_filter(xb * m, sg) / Gm
    return nulls(xh, kh, core)


say('')
say('%-18s %-10s %9s %9s %9s %9s %6s' % ('map', 'band', 'r', 'wrapmax', 'overmax', 'wrapmin', 'z'))
HP = {}
todo = [(k, a, ((192.0, 2), (48.0, 2))) for k, a in maps.items()] + [('ref direct 18-22', direct, ((192.0, 2), (48.0, 2), (192.0, 0), (48.0, 0)))]
for k, a, bands in todo:
    for su, b in bands:
        h = hp(a, su, b)
        h['z'] = (h['r'] - h['wrap_mean']) / h['wrap_sd']
        HP['%s|hp%d b%d' % (k, su, b)] = h
        say('%-18s hp%-3d b%d %+9.4f %+9.4f %+9.4f %+9.4f %6.1f' % (k, su, b, h['r'], h['wrap_max'], h['over_max'], h['wrap_min'], h['z']))
# gate 1 raw half
r0 = pear(direct[M], key[M])
wr = np.array([pear(direct[M & np.roll(M, s, (0, 1))], np.roll(key, s, (0, 1))[M & np.roll(M, s, (0, 1))]) for s in SHIFTS])
ov = np.array([crop_r(direct, key, M, dy, dx) for dy, dx in SHIFTS])
say('%-18s raw  b0 %+9.4f %+9.4f %+9.4f %+9.4f %6.1f' % ('ref direct 18-22', r0, wr.max(), ov.max(), wr.min(), (r0 - wr.mean()) / wr.std()))
HP['ref direct 18-22|raw b0'] = {'r': r0, 'wrap_max': float(wr.max()), 'over_max': float(ov.max()), 'wrap_min': float(wr.min()), 'over_min': float(ov.min())}

# gate 2
say('')
g2 = np.load(T + 'gate2/252405-ink9-s43-75k-c+0-rev.npy')
s43c0 = np.load(T + 'maps/s43_75k-c+0-rev.npy')
say('  gate-2 map vs maps/s43_75k-c+0-rev: max|d| %.3g' % np.abs(g2 - s43c0).max())


def scan(p, rad):
    ps, ks = gaussian_filter(p.astype(np.float64), 4), gaussian_filter(key, 4)
    out = np.full((2 * rad + 1, 2 * rad + 1), np.nan)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            out[dy + rad, dx + rad] = crop_r(ps, ks, M, dy, dx)
    return out


G2 = {}
for k, p in (('s43_75k c+0 rev (gate 2)', g2), ('arm tauil c+0 rev', np.load(T + 'maps/tauil_ft6k-c+0-rev.npy'))):
    sc = scan(p, 6)
    iy, ix = np.unravel_index(np.argmax(sc), sc.shape)
    near = sc[5:8, 5:8]
    say('  %-26s peak %+.5f at (%+d, %+d); r(0,0) %+.5f; best within 1 px %+.5f; gate rule %s'
        % (k, sc[iy, ix], iy - 6, ix - 6, sc[6, 6], near.max(), 'PASS' if (abs(iy - 6) <= 1 and abs(ix - 6) <= 1) else 'FAIL'))
    G2[k] = {'peak': float(sc[iy, ix]), 'at': [int(iy - 6), int(ix - 6)], 'r00': float(sc[6, 6]), 'best_within_1': float(near.max())}

# diagnostics: registration of key and arm against the HF and S3 label grids
say('')
say('  DIAGNOSTIC (not a rule): AUCsup with the map rolled by (dy, dx), -4..4 px')
DG = {}
for k, a, p, n in (('key vs HF', key, POS, NEG), ('key vs S3', key, POS3, NEG3), ('arm vs HF', maps['arm tauil rev'], POS, NEG),
                   ('arm vs S3', maps['arm tauil rev'], POS3, NEG3)):
    best = (-1, 0, 0)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            v = auc(np.roll(a, (dy, dx), (0, 1)), p, n)
            best = max(best, (v, dy, dx))
    v0 = auc(a, p, n)
    DG[k] = {'at0': v0, 'best': best[0], 'best_at': [best[1], best[2]]}
    say('    %-10s at (0,0) %.4f; best %.4f at (%+d, %+d)' % (k, v0, best[0], best[1], best[2]))

# decision line
say('')
A = rows['arm tauil rev']
h192 = HP['arm tauil rev|hp192 b2']
hp_ok = h192['r'] > max(h192['wrap_max'], h192['over_max'])
ctl = {}
for k in ('ctl fwd', 'ctl shuf21', 'ctl shuf41'):
    a, b = HP[k + '|hp192 b2'], HP[k + '|hp48 b2']
    ctl[k] = {'in192': a['r'] <= min(a['wrap_max'], a['over_max']), 'in48': b['r'] <= min(b['wrap_max'], b['over_max']),
              'in192_wrap': a['r'] <= a['wrap_max']}
    say('  control %-10s 192 b2 %s   48 b2 %s' % (k, 'inside' if ctl[k]['in192'] else 'OUTSIDE', 'inside' if ctl[k]['in48'] else 'OUTSIDE'))
g1 = HP['ref direct 18-22|raw b0']
g1_raw_in = g1['wrap_min'] <= g1['r'] <= g1['wrap_max'] and g1['r'] <= g1['over_max']
g1_hp48 = HP['ref direct 18-22|hp48 b0']; g1_hp192 = HP['ref direct 18-22|hp192 b0']
g1_pass = g1_raw_in and g1_hp48['r'] <= g1_hp48['wrap_max'] and g1_hp192['r'] <= g1_hp192['wrap_max']
g2_pass = all(abs(v) <= 1 for v in G2['s43_75k c+0 rev (gate 2)']['at'])
say('  gate 1: raw inside %s, hp48 b0 inside %s, hp192 b0 inside %s -> %s' % (g1_raw_in, g1_hp48['r'] <= g1_hp48['wrap_max'],
                                                                               g1_hp192['r'] <= g1_hp192['wrap_max'], 'PASS' if g1_pass else 'FAIL'))
say('  gate 2: %s' % ('PASS' if g2_pass else 'FAIL'))


def verdict(n, hp_ok, c_ok):
    if n >= 0.771:
        return 'GO' if (hp_ok and c_ok) else 'AMBIGUOUS'
    return 'AMBIGUOUS' if n >= 0.661 else 'EXHAUSTED'


readings = {
    'v2 literal, controls at 192 um (fwd + shuf21)': ctl['ctl fwd']['in192'] and ctl['ctl shuf21']['in192'],
    'controls at 192 um, all three': all(c['in192'] for c in ctl.values()),
    'controls at 48 um (gates/prereg.json band) fwd + shuf21': ctl['ctl fwd']['in48'] and ctl['ctl shuf21']['in48'],
    'controls at 192 and 48 um, all three (test part primary)': all(c['in192'] and c['in48'] for c in ctl.values()),
}
DEC = {}
for k, c_ok in readings.items():
    v = verdict(A['nauc'], hp_ok, c_ok)
    vg = v if (v != 'GO' or (g1_pass and g2_pass)) else 'AMBIGUOUS (sanity gate failed)'
    DEC[k] = [v, vg]
    say('  arm, %-58s decision line %-9s with gates %s' % (k, v, vg))
say('  baseline nAUC %.4f -> %s by nAUC alone' % (rows['base s43_75k rev']['nauc'], verdict(rows['base s43_75k rev']['nauc'], True, True)))

J.update({'rows': rows, 'hp': HP, 'gate2': G2, 'diag_shift_auc': DG, 'controls': ctl, 'gate1_pass': bool(g1_pass), 'gate2_pass': bool(g2_pass),
          'decision': DEC})
json.dump(J, open(os.path.splitext(OUT)[0] + '.json', 'w'), indent=1, default=float)
open(OUT, 'w').write('\n'.join(lines) + '\n')
say('done %s' % time.strftime('%H:%M:%S'))
