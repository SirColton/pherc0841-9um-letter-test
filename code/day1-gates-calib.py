"""day1-gates-calib.py - plan step 0, a calibration the prereg cites. How well does ScrollPrize's own 2.4 um map
(the hp r reference, model 20260417190342) score against each label set on w00 and on 252405? If the 2.4 um
reference itself sits near the 9 um decision bar on 252405, the bar means something different there.

Both label sets on ONE grid so they compare: the '_transformed' (HF v2) tifxyz grid, one value per vertex
(~48 um cells). The ScrollPrize map lives on the published 2.403 grid, so each HF vertex takes the map's mean
over the 20x20 cell of its 3D-nearest published vertex (NN distance reported). The S3 20260918 labels are
carried the same way. Labels per HF cell: ink fraction and supervision fraction (cell means).
AUC with supervised negatives: pos = ink frac >= 0.75; neg = ink frac == 0 and supervision frac >= 0.75.
CPU, local files only; prints, writes nothing. Nothing at 9 um is read.
"""
import os
import numpy as np, tifffile, zarr
from scipy.spatial import cKDTree

G = 'geom/day1-9um/gates/s3'
A = 'geom/l0-1203-render/audit/'
SEG = {'w00': ('20260220213127-w00', '20260220213127', A + 'evidence-control-training-labels/w00_%s.tif', A + 'evidence-control-training-labels/w00_w00_%s.tif'),
       '252405': ('20260221022814-auto_grown_20260220174252405', '20260221022814', A + 'design-use-what-exists/seg/auto_grown_20260220174252405_%s.tif',
                  A + 'design-use-what-exists/seg/auto_grown_20260220174252405_%s.tif')}

def cell_mean(m, shape, f):
    H, W = shape
    si = np.minimum(np.ceil(np.arange(H) * f).astype(int), m.shape[0] - 1)
    sj = np.minimum(np.ceil(np.arange(W) * f).astype(int), m.shape[1] - 1)
    ni = np.diff(np.append(si, m.shape[0])); nj = np.diff(np.append(sj, m.shape[1]))
    r = np.add.reduceat(m.astype(np.float64), si, axis=0)
    r = np.add.reduceat(r, sj, axis=1)
    return r / np.maximum(ni[:, None] * nj[None, :], 1)

def auc(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]); rk = v.argsort(kind='stable').argsort() + 1.0; n1, n0 = pos.sum(), neg.sum()
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

for n, (seg, sid, fx, fl) in SEG.items():
    d = os.path.join(G, seg, 'mesh', '%s-on-20260319124803-2.403um.tifxyz' % sid)
    Xs, Ys, Zs = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    oks = (Xs > 0) & (Ys > 0) & (Zs > 0)
    ink = tifffile.imread(os.path.join(G, seg, 'ink-detection', 'PHerc0841-%s-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif' % sid))
    fs = ink.shape[0] / Xs.shape[0]
    lb = os.path.join(G, seg, 'ink-labels', '2.403um-volume-20260319124803', '20260918')
    S3L = {k: zarr.open_group(os.path.join(lb, k + '.zarr'), mode='r')['0'][:] for k in ('inklabels', 'supervision')}
    m_s = cell_mean(ink, Xs.shape, fs)
    lab_s = cell_mean(S3L['inklabels'] > 0, Xs.shape, fs); sup_s = cell_mean(S3L['supervision'] > 0, Xs.shape, fs)
    cov_s = cell_mean(ink > 0, Xs.shape, fs)
    Xh, Yh, Zh = (tifffile.imread(fx % k).astype(np.float64) for k in 'xyz')
    okh = (Xh > 0) & (Yh > 0) & (Zh > 0)
    HL = {}
    for k, nm in (('inklabels', 'inklabels_v2'), ('supervision', 'supervision_mask_v2')):
        a = tifffile.imread(fl % nm); HL[k] = a.reshape(a.shape[-2:]) if a.ndim > 2 else a
    fh = HL['supervision'].shape[0] / Xh.shape[0]
    lab_h = cell_mean(HL['inklabels'] > 0, Xh.shape, fh); sup_h = cell_mean(HL['supervision'] > 0, Xh.shape, fh)
    # carry published-grid cell values to the HF vertices by 3D NN
    is_, js_ = np.nonzero(oks)
    tr = cKDTree(np.stack([Xs[oks], Ys[oks], Zs[oks]], 1))
    ih, jh = np.nonzero(okh)
    dd, kk = tr.query(np.stack([Xh[okh], Yh[okh], Zh[okh]], 1))
    near = dd < 20                                  # within one published grid step (20 vox, 48 um)
    mh = m_s[is_[kk], js_[kk]]; covh = cov_s[is_[kk], js_[kk]]
    labsh = lab_s[is_[kk], js_[kk]]; supsh = sup_s[is_[kk], js_[kk]]
    lh, sh = lab_h[ih, jh], sup_h[ih, jh]
    use = near & (covh > 0.99)
    print('%-7s HF grid %s  published grid %s  NN to published median %.2f vox, p99 %.2f; vertices used %d of %d' % (
        n, Xh.shape, Xs.shape, np.median(dd), np.percentile(dd, 99), use.sum(), okh.sum()))
    for setname, L_, S_ in (('HF v2 labels', lh, sh), ('S3 20260918 labels (carried)', labsh, supsh)):
        pos = use & (L_ >= 0.75); neg = use & (L_ <= 0) & (S_ >= 0.75); negall = use & (L_ <= 0)
        print('   %-30s pos %6d neg(sup) %6d  AUC sup-neg %.4f   | neg(all) %6d  AUC all-neg %.4f' % (
            setname, pos.sum(), neg.sum(), auc(mh, pos, neg), negall.sum(), auc(mh, pos, negall)))
    agree = use & (sh >= 0.75) & (supsh >= 0.75)
    print('   label agreement on co-supervised cells: n %d, Pearson r of ink fractions %.3f' % (agree.sum(), np.corrcoef(lh[agree], labsh[agree])[0, 1]))
