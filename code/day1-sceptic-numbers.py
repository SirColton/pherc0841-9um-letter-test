"""day1-sceptic-numbers.py - sceptic re-check of the day-1 headline numbers, own code, CPU only, cached files only.

Writes only geom/day1-9um/sceptic-numbers/sceptic-run.txt (and prints). No model runs; no 9 um model output is
read on 252405 (none exists); on 252405 only the model-free direct render and the ScrollPrize 2.4 um map are used.

AUC here is an independent implementation (searchsorted counts, ties = 1/2), not rankdata.
hp r is implemented twice:
  B = the baselines part's convention (np.roll with wrap, core = M & G(M,2s) > 0.999, no read>0 term);
  P = prereg.json's text (overlap-only shifts, core also needs read > 0).
Keys on w00big: 'ds8' = the on-disk ds8 jpg, bilinear exactly as day1-score-baselines.py does;
                'tif' = render/w00big-republished/w00big-spmap-grid.npy (full-res tif, 4x4 block mean, by index),
                which is what prereg.json says selection uses.
Sigma 48 um (prereg) and 192 um (AMENDMENT-1 item 1).
"""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

OUT = 'geom/day1-9um/sceptic-numbers/sceptic-run.txt'
V = 'geom/l0-1203-render/audit/verify-nine-um-test-does-it-matter'
B = 'geom/l0-1203-render/audit/build-deciding-test'
C24 = 'geom/c0841/mesh0b-render/w00big'
MAPS = 'geom/day1-9um/baselines/maps'
REP = 'geom/day1-9um/render/w00big-republished'
R = 'geom/day1-9um/render'
H, W, P, E = 995, 1245, 4, 40
PX = 2.403 * P
SH = [(dy, dx) for dy in (-312, -208, -104, 0, 104, 208, 312) for dx in (-312, -208, -104, 0, 104, 208, 312) if (dy, dx) != (0, 0)]
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


def auc(s, pos, neg):
    p = np.sort(s[pos].astype(np.float64)); n = np.sort(s[neg].astype(np.float64))
    lo = np.searchsorted(n, p, 'left'); hi = np.searchsorted(n, p, 'right')
    return float((lo + 0.5 * (hi - lo)).sum() / (len(p) * len(n)))


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ---------------- w00big frame ----------------
lab = np.load(os.path.join(V, 'w00_inklabels_v2-grid.npy')); sup = np.load(os.path.join(V, 'w00_supervision_mask_v2-grid.npy'))
mask = np.load(os.path.join(B, 'ours-mask.npy'))
m24 = np.asarray(Image.open(os.path.join(C24, 'mask.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).min((1, 3)) > 0
VAL = mask & m24
M = VAL.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
pos = M & (lab >= 0.75); negS = M & (lab <= 0) & (sup >= 0.75); negA = M & (lab <= 0)
say('w00big: scored %d  pos %d  negS %d  negA %d' % (M.sum(), pos.sum(), negS.sum(), negA.sum()))

ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
gi = 2.5 * (380 + (np.arange(H) * P + (P - 1) / 2) / 20.0) - 0.5
gj = 2.5 * (500 + (np.arange(W) * P + (P - 1) / 2) / 20.0) - 0.5
GJ, GI = np.meshgrid(gj, gi)
ds8 = map_coordinates(ink, [GI, GJ], order=1).astype(np.float32)
tif = np.load(os.path.join(REP, 'w00big-spmap-grid.npy')).astype(np.float32)
h24 = np.asarray(Image.open(os.path.join(C24, 'hecate-rev.png')), np.float32)[:H * P, :W * P].reshape(H, P, W, P).mean((1, 3))
labS3 = np.load(os.path.join(REP, 'w00big-inklabels-grid.npy')); supS3 = np.load(os.path.join(REP, 'w00big-supervision-grid.npy'))
posS3 = M & (labS3 >= 0.75); negS3 = M & (labS3 <= 0) & (supS3 >= 0.75)


def mblur(x, s):
    if not s:
        return x
    w = gaussian_filter(VAL.astype(np.float32), s)
    return gaussian_filter(np.where(VAL, x, 0).astype(np.float32), s) / np.maximum(w, 1e-3)


def hpf(x, sig):
    m = M.astype(np.float32)
    return x - gaussian_filter(x * m, sig) / np.maximum(gaussian_filter(m, sig), 1e-3)


CACHE = {}


def keyprep(kname, sig):
    k = (kname, sig)
    if k not in CACHE:
        key = {'ds8': ds8, 'tif': tif, 'h24': h24}[kname]
        core = M & (gaussian_filter(M.astype(np.float32), 2 * sig) > 0.999)
        CACHE[k] = (hpf(key, sig), core)
    return CACHE[k]


def hpr(x, kname, sig, blur, conv):
    kh, core = keyprep(kname, sig)
    xh = hpf(mblur(np.asarray(x, np.float32), blur), sig)
    c = core & (x > 0) if conv == 'P' else core
    r0 = pear(xh[c], kh[c])
    nl = []
    for dy, dx in SH:
        if conv == 'B':
            kr = np.roll(kh, (dy, dx), (0, 1)); cr = core & np.roll(core, (dy, dx), (0, 1))
            nl.append(pear(xh[cr], kr[cr]))
        else:
            ys, ye = max(0, dy), min(H, H + dy); xs, xe = max(0, dx), min(W, W + dx)
            a = xh[ys:ye, xs:xe]; b = kh[ys - dy:ye - dy, xs - dx:xe - dx]
            mm = c[ys:ye, xs:xe] & core[ys - dy:ye - dy, xs - dx:xe - dx]
            nl.append(pear(a[mm], b[mm]))
    nl = np.array(nl)
    return r0, float(nl.max()), float((r0 - nl.mean()) / nl.std()), int((nl >= r0).sum())


def rnorm_prereg(a):
    r = np.full(a.shape, 0.5, np.float32)
    v = a[M].astype(np.float64)
    order = np.argsort(v, kind='stable'); sv = v[order]
    # average ranks for ties, 1-based
    _, first, cnt = np.unique(sv, return_index=True, return_counts=True)
    avg = first + (cnt + 1) / 2.0
    rk = np.empty_like(v); rk[order] = np.repeat(avg, cnt)
    r[M] = (rk - 0.5) / len(v)
    return r


def ld(model, c, o):
    return np.load(os.path.join(MAPS, '%s-c%+d-%s.npy' % (model, c, o))).astype(np.float32)


C5 = (-4, -2, 0, 2, 4)
mean5 = {(m, o): np.mean([ld(m, c, o) for c in C5], 0) for m, o in (('s43_75k', 'rev'), ('soup42', 'rev'), ('soup43', 'rev'),
                                                                        ('soup42', 'fwd'), ('soup43', 'fwd'), ('s43_50k', 'rev'))}
R4 = np.mean([rnorm_prereg(mean5[('soup42', 'rev')]), rnorm_prereg(mean5[('soup43', 'rev')])], 0)
stack = np.load(os.path.join(B, 'ours-stack.npy'), mmap_mode='r')
direct = np.asarray(stack[18:23], np.float32).mean(0)

# reproduction check of maps-check claim
a1 = np.load(os.path.join(B, 'ink9-ours-c+0-rev.npy')).astype(np.float32)
a2 = np.load('geom/day1-9um/baselines/maps-check/s43_50k-c+0-rev.npy').astype(np.float32)
a3 = ld('s43_50k', 0, 'rev')
say('s43_50k c+0 rev: max|maps-check - ink9-ours-c+0-rev| %.6g ; max|maps - ink9-ours| %.6g' % (np.abs(a1 - a2).max(), np.abs(a1 - a3).max()))
say('s43_50k mean3(-4,0,+4) from maps: AUCsup %.4f ; c+0 AUCsup %.4f' % (
    auc(np.mean([ld('s43_50k', c, 'rev') for c in (-4, 0, 4)], 0), pos, negS), auc(a3, pos, negS)))
say('')

ROWS = [('R1 cand s43_75k rev mean5', mean5[('s43_75k', 'rev')]), ('R1 cand soup42 rev mean5', mean5[('soup42', 'rev')]),
        ('R1 cand soup43 rev mean5', mean5[('soup43', 'rev')]), ('R4-ondisk rank-mean soup42+soup43', R4),
        ('extra s43_50k rev mean5', mean5[('s43_50k', 'rev')]),
        ('ctrl soup42 fwd mean5', mean5[('soup42', 'fwd')]), ('ctrl soup43 fwd mean5', mean5[('soup43', 'fwd')]),
        ('ref direct render 18-22', direct), ('ref Hecate 2.4 rev', h24), ('ref SP ds8', ds8), ('ref SP tif', tif)]
say('%-36s %7s %7s %7s | hp r (r / nullmax / n>=r)' % ('map', 'AUCsup', 'AUCall', 'AUCsupS3'))
res = {}
for name, a in ROWS:
    row = {'auc': auc(a, pos, negS), 'aucA': auc(a, pos, negA), 'aucS3': auc(a, posS3, negS3)}
    say('%-36s %7.4f %7.4f %7.4f' % (name, row['auc'], row['aucA'], row['aucS3']))
    for kname in ('ds8', 'tif'):
        for sig_um in (48.0, 192.0):
            for conv in ('B', 'P'):
                parts = []
                for b in (0, 2, 4):
                    r0, mx, z, nge = hpr(a, kname, sig_um / PX, b, conv)
                    row[(kname, sig_um, conv, b)] = (r0, mx, z, nge)
                    parts.append('b%d %+.3f/%+.3f/%d' % (b, r0, mx, nge))
                say('     key %-3s sig %3d um conv %s: %s' % (kname, sig_um, conv, '  '.join(parts)))
    res[name] = row
say('')
say('SELECTION under prereg.json rule (AUCsup HF v2 band 0.01, then highest hp r at blur 2), recipes R1 cands + R4-ondisk:')
rec = [n for n, _ in ROWS[:4]]
best = max(res[n]['auc'] for n in rec)
band = [n for n in rec if res[n]['auc'] >= best - 0.01]
say('  best AUCsup %.4f ; in band: %s' % (best, band))
for kname in ('ds8', 'tif'):
    for sig_um in (48.0, 192.0):
        for conv in ('B', 'P'):
            pick = max(band, key=lambda n: res[n][(kname, sig_um, conv, 2)][0])
            say('  key %s sig %d conv %s -> %s  (%s)' % (kname, sig_um, conv, pick,
                ', '.join('%s %.3f' % (n.split()[-2] if 'R1' in n else 'R4', res[n][(kname, sig_um, conv, 2)][0]) for n in band)))
say('')

# ---------------- 252405: model-free only ----------------
m5 = np.load(os.path.join(R, '252405-mask.npy'))
sp5 = np.load(os.path.join(R, '252405-spmap-grid.npy')).astype(np.float32)
l5 = np.load(os.path.join(R, '252405-inklabels-grid.npy')); s5 = np.load(os.path.join(R, '252405-supervision-grid.npy'))
st5 = np.load(os.path.join(R, '252405-stack.npy'), mmap_mode='r')
d5 = np.asarray(st5[18:23], np.float32).mean(0)
H5, W5 = m5.shape
M5 = m5.copy(); M5[:E] = M5[-E:] = False; M5[:, :E] = M5[:, -E:] = False
say('252405 render grid %dx%d, mask %d, scored after %d px edge cut %d (%.3f cm2)' % (H5, W5, m5.sum(), E, M5.sum(), M5.sum() * (PX * 1e-4) ** 2))


def shifted_r(p, q, Mk, dy, dx):
    ys, ye = max(0, dy), min(H5, H5 + dy); xs, xe = max(0, dx), min(W5, W5 + dx)
    mm = Mk[ys:ye, xs:xe] & Mk[ys - dy:ye - dy, xs - dx:xe - dx]
    return pear(p[ys:ye, xs:xe][mm], q[ys - dy:ye - dy, xs - dx:xe - dx][mm])


r0 = pear(d5[M5], sp5[M5]); nl = np.array([shifted_r(d5, sp5, M5, dy, dx) for dy, dx in SH])
say('  GATE 1 raw r (layers 18-22 mean vs SP map): r %+.4f  null mean %+.4f max %+.4f  z %.2f  n>=r %d' % (
    r0, nl.mean(), nl.max(), (r0 - nl.mean()) / nl.std(), (nl >= r0).sum()))
for sig_um in (48.0, 192.0):
    sig = sig_um / PX
    m = M5.astype(np.float32); g = np.maximum(gaussian_filter(m, sig), 1e-3)
    dh = d5 - gaussian_filter(d5 * m, sig) / g; kh = sp5 - gaussian_filter(sp5 * m, sig) / g
    core = M5 & (gaussian_filter(m, 2 * sig) > 0.999) & (d5 > 0)
    h0 = pear(dh[core], kh[core]); hn = np.array([shifted_r(dh, kh, core, dy, dx) for dy, dx in SH])
    say('  GATE 1 hp r blur 0 (prereg half), sigma %d um: r %+.4f  null mean %+.4f max %+.4f  z %.2f  n>=r %d  core %d' % (
        sig_um, h0, hn.mean(), hn.max(), (h0 - hn.mean()) / hn.std(), (hn >= h0).sum(), core.sum()))
p5 = m5 & (l5 >= 0.75); n5 = m5 & (l5 <= 0) & (s5 >= 0.75); a5 = m5 & (l5 <= 0)
say('  calib on the render grid (SP map vs S3 20260918 labels, 9.6 um blocks, render mask, no edge cut):')
say('    pos %d negS %d AUCsup %.4f | negAll %d AUCall %.4f' % (p5.sum(), n5.sum(), auc(sp5, p5, n5), a5.sum(), auc(sp5, p5, a5)))
say('    same with the 40 px edge cut: pos %d negS %d AUCsup %.4f' % ((p5 & M5).sum(), (n5 & M5).sum(), auc(sp5, p5 & M5, n5 & M5)))
say('    ink-labelled area on the render grid (frac>=0.5 blocks): %.4f cm2; sum of fractions %.4f cm2' % (
    (m5 & (l5 >= 0.5)).sum() * (PX * 1e-4) ** 2, (l5 * m5).sum() * (PX * 1e-4) ** 2))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
