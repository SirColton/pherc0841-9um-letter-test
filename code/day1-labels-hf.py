"""day1-labels-hf.py - prereg v2's decisive labels for 252405: ScrollPrize's HF inklabels_v2,
supervision_mask_v2 and validation_mask_v2 carried onto the 252405 render grid by the w00 convention,
their agreement with the S3 20260918 grids, and the ceiling C the normalised AUC divides by.

WHY. prereg-v2.json (sha256 49633d68...) makes the decisive 252405 labels the HF v2 set carried "by the
same convention as w00 (rebuild; the existing grids are S3 20260918, reported beside, never decisive)",
and fixes nAUC = (AUCsup - 0.5) / (C - 0.5) with C = AUCsup of ScrollPrize's 2.4 um map "on the same
labels and pixels". The 0.8609 it quotes was measured on the HF tifxyz's own ~48 um cells over the
whole segment (day1-gates-calib.py), not on the render grid; this measures C where the 9 um maps will
be scored.

THE CONVENTION, labels-on-grid.py's (verify-nine-um-test-does-it-matter), unchanged: render grid point
(i, j) = bilinear position on the render mesh at ((4 i + 1.5) / 20, (4 j + 1.5) / 20) cells of the crop
-> 3D nearest point of the HF segment's own tifxyz densified 5 sub-samples per cell (a sub-sample kept
only when its four corner cells are valid) -> raster px = that sub-sample's (row, col) x 20 -> mean of
the 4 x 4 raster px at rint(R - 1.5 + a), rint(C - 1.5 + b), / 255. No distance limit (the convention
has none; the distance is reported). The one generalisation: labels-on-grid.py's hand-picked dataset
window becomes the dataset cells inside the grid points' bounding box +- 60 voxels (+-2 cells),
day1-render-labelcheck.py's rule; it is checked by rebuilding the w00 grid on the w00big frame and
comparing it with the saved w00_inklabels_v2-grid.npy element by element.

SCORED PIXELS (prereg silent; conservative reading, marked): the render mask with day1's 40 px edge cut
(day1-render-9um.py gate 1, day1-score-baselines.py), 4,468,325 px; ScrollPrize's map is non-zero on
every one of them, so no coverage cut applies. pos = label >= 0.75; neg = label <= 0 AND supervision >=
0.75 (selection_on_w00big_only.primary). AUC by midranks (day1-score-baselines.py). Validation is
carried and reported; the primary C does not restrict to it, because the primary rule names only
labels and supervision (a validation-restricted C is printed beside, not used).

FETCH: the seven files come from the HF bucket scrollprize/datasets, ink/841/auto_grown_20260220174252405/
(the bucket the w00 dataset came from, ink/841/w00/), into <out>/src/; each is compared with the copy in
design-use-what-exists/seg (sha256) and with the saved tree listing's xetHash (from the redirect).

Usage: day1-labels-hf.py <out dir>     CPU only (no model); writes only into <out dir>. Run from <workdir>.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np
import tifffile
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import rankdata

import runfile

OUTD = sys.argv[1]
os.makedirs(os.path.join(OUTD, 'src'), exist_ok=True)
OUT = os.path.join(OUTD, 'labels-hf-run.txt')
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


A = 'geom/l0-1203-render/audit/'
SEG = 'auto_grown_20260220174252405'
LOCAL = A + 'design-use-what-exists/seg/' + SEG + '_'
TREE = A + 'sota-research/tree-841_' + SEG + '.json'
HF = 'https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/' + SEG + '/'
FILES = {SEG + '_inklabels_v2.tif': 'inklabels_v2.tif', SEG + '_supervision_mask_v2.tif': 'supervision_mask_v2.tif',
         SEG + '_validation_mask_v2.tif': 'validation_mask_v2.tif', 'x.tif': 'x.tif', 'y.tif': 'y.tif', 'z.tif': 'z.tif',
         'meta.json': 'meta.json'}
R = 'geom/day1-9um/render/'
MESH, CROP = R + 'src/mesh-2.403', (42, 468, 210, 664)
POOL, SUB, E = 4, 5, 40


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def xet_of(url):
    op = urllib.request.build_opener(NoRedirect)
    try:
        op.open(urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'scrolls-fl'}), timeout=60)
    except urllib.error.HTTPError as e:
        loc = e.headers.get('Location', '')
        return loc.split('?')[0].rsplit('/', 1)[-1] if e.code in (301, 302, 303, 307, 308) else 'HTTP %d' % e.code
    return 'no redirect'


say('day1-labels-hf.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
say('  prereg-v2.json sha256 %s' % sha256('geom/day1-9um/prereg-v2.json'))
tree = {e['path'].rsplit('/', 1)[-1]: e for e in json.load(open(TREE)) if e['type'] == 'file'}
say('')
say('  FETCH  %s  (HF bucket scrollprize/datasets)' % HF)
src = {}
for remote, short in FILES.items():
    dst = os.path.join(OUTD, 'src', remote)
    xet = xet_of(HF + remote)
    if not os.path.exists(dst):
        with urllib.request.urlopen(urllib.request.Request(HF + remote, headers={'User-Agent': 'scrolls-fl'}), timeout=300) as r, open(dst + '.part', 'wb') as f:
            f.write(r.read())
        os.replace(dst + '.part', dst)
    loc = LOCAL + short
    s_new, s_loc = sha256(dst), (sha256(loc) if os.path.exists(loc) else None)
    t = tree.get(remote, {})
    say('    %-48s %9d B  sha256 %s  | local seg copy %s | xet live %s listing %s'
        % (remote, os.path.getsize(dst), s_new[:16], 'SAME' if s_new == s_loc else ('DIFFERENT' if s_loc else 'absent'),
           xet[:16], 'SAME' if xet == t.get('xetHash') else 'DIFFERENT (%s)' % str(t.get('xetHash'))[:16]))
    src[short] = {'path': dst, 'sha256': s_new, 'bytes': os.path.getsize(dst), 'same_as_local_seg': s_new == s_loc,
                  'xet_live': xet, 'xet_listing': t.get('xetHash')}


def bil(A_, ii, jj):
    i1 = np.clip(np.floor(ii).astype(int), 0, A_.shape[0] - 2)
    j1 = np.clip(np.floor(jj).astype(int), 0, A_.shape[1] - 2)
    di, dj = ii - i1, jj - j1
    return (A_[i1, j1] * (1 - di) * (1 - dj) + A_[i1 + 1, j1] * di * (1 - dj)
            + A_[i1, j1 + 1] * (1 - di) * dj + A_[i1 + 1, j1 + 1] * di * dj)


def grid_points(X, Y, Z):
    h, w = X.shape
    H, W = (h - 1) * 20 // POOL, (w - 1) * 20 // POOL
    ii = (np.arange(H) * POOL + (POOL - 1) / 2) / 20.0
    jj = (np.arange(W) * POOL + (POOL - 1) / 2) / 20.0
    JJ, II = np.meshgrid(jj, ii)
    return np.stack([bil(A_, II, JJ) for A_ in (X, Y, Z)], -1)


def carry(Q, X, Y, Z, rasters, qmask=None):
    """labels-on-grid.py's carry; the window is the dataset cells inside Q's bounding box +- 60 voxels."""
    q = Q.reshape(-1, 3) if qmask is None else Q[qmask]
    lo, hi = q.min(0) - 60, q.max(0) + 60
    ok0 = (X > 0) & (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (Z >= lo[2]) & (Z <= hi[2])
    rr, cc = np.nonzero(ok0)
    r0, r1, c0, c1 = max(rr.min() - 2, 0), min(rr.max() + 3, X.shape[0]), max(cc.min() - 2, 0), min(cc.max() + 3, X.shape[1])
    RR_, CC_ = np.arange(r0, r1 - 1, 1.0 / SUB), np.arange(c0, c1 - 1, 1.0 / SUB)
    CC, RR = np.meshgrid(CC_, RR_)
    i1, j1 = np.floor(RR).astype(int), np.floor(CC).astype(int)
    ok = (X[i1, j1] > 0) & (X[i1 + 1, j1] > 0) & (X[i1, j1 + 1] > 0) & (X[i1 + 1, j1 + 1] > 0)
    P = np.stack([bil(A_, RR, CC)[ok] for A_ in (X, Y, Z)], 1)
    dr, dc = RR[ok], CC[ok]
    del RR, CC, i1, j1
    T = cKDTree(P)
    dist, idx = T.query(Q.reshape(-1, 3), workers=4)
    Rr, Cr = dr[idx].reshape(Q.shape[:2]) * 20, dc[idx].reshape(Q.shape[:2]) * 20
    edge = ((dr[idx] <= r0 + 1e-6) | (dr[idx] >= r1 - 1 - 1.0 / SUB - 1e-6) | (dc[idx] <= c0 + 1e-6) | (dc[idx] >= c1 - 1 - 1.0 / SUB - 1e-6)).reshape(Q.shape[:2])
    out = {}
    for name, L in rasters.items():
        acc = np.zeros(Q.shape[:2], np.float32)
        for a in range(4):
            for b in range(4):
                ri = np.clip(np.rint(Rr - 1.5 + a).astype(int), 0, L.shape[0] - 1)
                ci = np.clip(np.rint(Cr - 1.5 + b).astype(int), 0, L.shape[1] - 1)
                acc += L[ri, ci] / 255.0
        out[name] = acc / 16
    return out, dist.reshape(Q.shape[:2]), (r0, r1, c0, c1), len(P), edge


def tif2(p):
    a = tifffile.imread(p)
    return a.reshape(a.shape[-2:]) if a.ndim > 2 else a


# ---- convention check: rebuild w00's grid on the w00big frame ------------------------------------
say('')
V = A + 'verify-nine-um-test-does-it-matter/'
DS = A + 'sota-research/w00-dataset/'
x, y, z = (np.array(Image.open('geom/c0841/mesh0b/w00big/' + k + '.tif'), np.float64) for k in 'xyz')
Qw = grid_points(x, y, z)
Xd, Yd, Zd = (tifffile.imread(DS + k + '.tif').astype(np.float64) for k in 'xyz')
cw, dw, win, npts, edw = carry(Qw, Xd, Yd, Zd, {'w00_inklabels_v2': tif2(DS + 'w00_inklabels_v2.tif'),
                                                 'w00_supervision_mask_v2': tif2(DS + 'w00_supervision_mask_v2.tif')})
conv = {}
for k, g in cw.items():
    old = np.load(V + k + '-grid.npy')
    d = np.abs(g - old)
    conv[k] = {'max_abs_diff': float(d.max()), 'px_differing': int((d > 1e-6).sum()), 'n': int(d.size)}
    say('  CONVENTION CHECK on w00big %s: grid %s; rebuilt vs saved %s-grid.npy: max |diff| %.3g, px differing %d of %d'
        % (k, g.shape, k, d.max(), (d > 1e-6).sum(), d.size))
del Xd, Yd, Zd, cw, Qw

# ---- 252405 --------------------------------------------------------------------------------------
X, Y, Z = (tifffile.imread(os.path.join(MESH, k + '.tif')).astype(np.float64)[CROP[0]:CROP[1], CROP[2]:CROP[3]] for k in 'xyz')
Q = grid_points(X, Y, Z)
H, W = Q.shape[:2]
mask = np.load(R + '252405-mask.npy')
assert mask.shape == (H, W), (mask.shape, H, W)
M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
Xh, Yh, Zh = (tifffile.imread(src[k + '.tif']['path']).astype(np.float64) for k in 'xyz')
ras = {'inklabels_v2': tif2(src['inklabels_v2.tif']['path']), 'supervision_mask_v2': tif2(src['supervision_mask_v2.tif']['path']),
       'validation_mask_v2': tif2(src['validation_mask_v2.tif']['path'])}
meta = json.load(open(src['meta.json']['path']))
grids, dist, win, npts, edge = carry(Q, Xh, Yh, Zh, ras, qmask=mask)
say('')
say('  252405: render grid %dx%d (mesh %s crop %s); HF tifxyz %s, raster %s (%.3f x %.3f px per cell; meta scale %s); factor 20 (the convention)'
    % (H, W, MESH, list(CROP), Xh.shape, ras['inklabels_v2'].shape, ras['inklabels_v2'].shape[0] / Xh.shape[0],
       ras['inklabels_v2'].shape[1] / Xh.shape[1], meta.get('scale')))
say('    dataset window rows %d-%d cols %d-%d, %d densified points; render mask %d px, scored (40 px edge cut) %d px (%.2f cm2)'
    % (win[0], win[1], win[2], win[3], npts, mask.sum(), M.sum(), M.sum() * (2.403 * POOL * 1e-4) ** 2))
dm = dist[M]
say('    grid -> HF surface nearest distance over the scored px (2.403 um voxels): median %.2f p95 %.2f p99 %.2f max %.2f; > 10 vox %d px, > 20 vox %d px; NN on the window edge %d px'
    % (np.median(dm), np.percentile(dm, 95), np.percentile(dm, 99), dm.max(), (dm > 10).sum(), (dm > 20).sum(), int((edge & M).sum())))
np.save(os.path.join(OUTD, 'nn-dist.npy'), dist.astype(np.float32))
out_sha = {}
for k, g in grids.items():
    p = os.path.join(OUTD, '252405_%s-grid.npy' % k)
    np.save(p, g.astype(np.float32))
    out_sha[os.path.basename(p)] = sha256(p)
    Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).save(p[:-4] + '.png')
    say('    %-22s over scored px: mean %.4f, >= 0.75 %.4f, <= 0 %.4f' % (k, g[M].mean(), (g[M] >= 0.75).mean(), (g[M] <= 0).mean()))

# ---- agreement with the S3 20260918 grids ---------------------------------------------------------
lab, sup, val = grids['inklabels_v2'], grids['supervision_mask_v2'], grids['validation_mask_v2']
S3 = {k: np.load(R + '252405-%s-grid.npy' % k) for k in ('inklabels', 'supervision', 'validation')}


def r_at(a, b, m, dy, dx):
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = m[ys:ye, xs:xe] & m[ys - dy:ye - dy, xs - dx:xe - dx]
    aa, bb = a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm]
    if aa.std() == 0 or bb.std() == 0:
        return -2.0
    return float(np.corrcoef(aa, bb)[0, 1])


def scan(a, b, m, Rr=30):
    r00 = r_at(a, b, m, 0, 0)
    best = max((r_at(a, b, m, dy, dx), dy, dx) for dy in range(-Rr, Rr + 1, 3) for dx in range(-Rr, Rr + 1, 3))
    best = max((r_at(a, b, m, dy, dx), dy, dx) for dy in range(best[1] - 3, best[1] + 4) for dx in range(best[2] - 3, best[2] + 4))
    return r00, best


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, a.sum() + b.sum()))


agree = {}
say('')
say('  AGREEMENT, HF v2 (decisive) vs S3 20260918 (by index, render/252405-*-grid.npy), scored px %d:' % M.sum())
for hk, sk, g in (('inklabels_v2', 'inklabels', lab), ('supervision_mask_v2', 'supervision', sup), ('validation_mask_v2', 'validation', val)):
    s = S3[sk]
    a75, b75 = M & (g >= 0.75), M & (s >= 0.75)
    r00 = float(np.corrcoef(g[M], s[M])[0, 1]) if s[M].std() > 0 and g[M].std() > 0 else float('nan')
    agree[hk] = {'r': r00, 'hf_frac_ge75': float(a75.sum() / M.sum()), 's3_frac_ge75': float(b75.sum() / M.sum()),
                 'dice_ge75': dice(a75, b75), 'both': int((a75 & b75).sum()), 'hf_only': int((a75 & ~b75).sum()), 's3_only': int((~a75 & b75).sum())}
    say('    %-20s r %+.4f | >= 0.75: HF %.4f S3 %.4f of px; both %d, HF only %d, S3 only %d; Dice %.4f'
        % (hk, r00, a75.sum() / M.sum(), b75.sum() / M.sum(), (a75 & b75).sum(), (a75 & ~b75).sum(), (~a75 & b75).sum(), dice(a75, b75)))
cos = M & (sup >= 0.75) & (S3['supervision'] >= 0.75)
r_cos = float(np.corrcoef(lab[cos], S3['inklabels'][cos])[0, 1])
r00, best = scan(lab, S3['inklabels'], M)
agree['ink_co_supervised'] = {'n': int(cos.sum()), 'r': r_cos}
agree['ink_shift_scan'] = {'r00': r00, 'best_r': best[0], 'best_dy': best[1], 'best_dx': best[2]}
say('    ink fraction on co-supervised px (both supervision >= 0.75): n %d, r %+.4f' % (cos.sum(), r_cos))
say('    ink grid shift scan (+-30 px; shift of the S3 grid that best matches HF): r(0,0) %+.4f, peak %+.4f at (%+d, %+d) render px (%.0f, %.0f um)'
    % (r00, best[0], best[1], best[2], best[1] * 9.612, best[2] * 9.612))
cls = lambda L_, S_: np.where(L_ >= 0.75, 1, np.where((L_ <= 0) & (S_ >= 0.75), 0, -1))
ch, cs = cls(lab, sup)[M], cls(S3['inklabels'], S3['supervision'])[M]
tab = {('pos', 'neg', 'other')[i] + '|' + ('pos', 'neg', 'other')[j]: int(((ch == (1, 0, -1)[i]) & (cs == (1, 0, -1)[j])).sum()) for i in range(3) for j in range(3)}
agree['class_table_hf_by_s3'] = tab
say('    scoring class table (rows HF, cols S3; pos = label >= 0.75, neg = label <= 0 & supervision >= 0.75, other = unscored):')
say('                 S3 pos    S3 neg  S3 other')
for i, n in enumerate(('pos', 'neg', 'other')):
    say('      HF %-5s %9d %9d %9d' % (n, tab[n + '|pos'], tab[n + '|neg'], tab[n + '|other']))

# ---- C: AUCsup of ScrollPrize's 2.4 um map -------------------------------------------------------
sp = np.load(R + '252405-spmap-grid.npy')


def auc(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]).astype(np.float64)
    rk = rankdata(v)
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ord(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]).astype(np.float64)
    rk = v.argsort(kind='stable').argsort() + 1.0
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


pos = M & (lab >= 0.75); negS = M & (lab <= 0) & (sup >= 0.75); negA = M & (lab <= 0)
C = auc(sp, pos, negS)
res = {'C': C, 'C_ordinal': auc_ord(sp, pos, negS), 'AUCall': auc(sp, pos, negA), 'pos': int(pos.sum()), 'neg_sup': int(negS.sum()), 'neg_all': int(negA.sum())}
Vm = val >= 0.75
res['C_in_validation'] = auc(sp, pos & Vm, negS & Vm)
res['val_pos'], res['val_neg_sup'] = int((pos & Vm).sum()), int((negS & Vm).sum())
pos3 = M & (S3['inklabels'] >= 0.75); neg3 = M & (S3['inklabels'] <= 0) & (S3['supervision'] >= 0.75)
res['C_s3'] = auc(sp, pos3, neg3)
res['s3_pos'], res['s3_neg_sup'] = int(pos3.sum()), int(neg3.sum())
res['bar_go_raw'] = 0.5 + 0.771 * (C - 0.5)
res['bar_ambiguous_raw'] = 0.5 + 0.661 * (C - 0.5)
say('')
say("  C = AUCsup of ScrollPrize's 2.4 um map (render/252405-spmap-grid.npy, full-res inkmap.tif by index) on the HF v2 grids, scored px:")
say('    pos %d  neg(sup) %d  -> C = %.4f (midrank; ordinal %.4f)   | neg(all) %d -> AUCall %.4f'
    % (res['pos'], res['neg_sup'], C, res['C_ordinal'], res['neg_all'], res['AUCall']))
say('    prereg quoted 0.8609 (HF tifxyz ~48 um cells, whole segment; day1-gates-calib.py); this C replaces it (task: "use YOURS")')
say('    beside, not decisive: inside HF validation_mask_v2 >= 0.75 only: pos %d neg(sup) %d -> %.4f; S3 20260918 grids, same px: pos %d neg(sup) %d -> %.4f'
    % (res['val_pos'], res['val_neg_sup'], res['C_in_validation'], res['s3_pos'], res['s3_neg_sup'], res['C_s3']))
say('    the bars in raw AUCsup with this C: nAUC 0.771 -> %.4f ; nAUC 0.661 -> %.4f' % (res['bar_go_raw'], res['bar_ambiguous_raw']))

json.dump({'prereg_sha256': sha256('geom/day1-9um/prereg-v2.json'), 'sources': src, 'convention_check_w00big': conv,
           'grid': {'H': H, 'W': W, 'mesh': MESH, 'crop': list(CROP), 'pool': POOL, 'sub': SUB, 'factor': 20, 'edge_cut': E,
                    'n_mask': int(mask.sum()), 'n_scored': int(M.sum()), 'dataset_window': [int(v) for v in win], 'n_points': int(npts),
                    'nn_dist_median': float(np.median(dm)), 'nn_dist_p99': float(np.percentile(dm, 99)), 'nn_dist_max': float(dm.max())},
           'outputs_sha256': out_sha, 'agreement_vs_s3': agree, 'ceiling': res,
           'pixels': 'render mask minus a 40 px edge cut; pos label>=0.75; neg label<=0 & supervision>=0.75; midrank AUC'},
          open(os.path.join(OUTD, 'labels-hf.json'), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
