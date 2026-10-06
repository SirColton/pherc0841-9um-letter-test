"""day1-gates-provenance.py - plan step 0, gates (a) and (b). CPU only, reads files already fetched into
geom/day1-9um/gates/ (prizeEligibility.json from ScrollPrize GitHub, catalog-live.json and s3/ meshes from
ScrollPrize's open-data bucket). Prints; writes nothing.

(a) Are PHerc1203 (20250820131727) and PHerc0846A (20250728152254) still on the First Letters list?
(b) Was w00 grown on the 113 keV 9.36 um scan? Four independent readings:
    1. the catalog's own segment record (original_volume_id, derived_from, source path);
    2. the catalog's scan records: which PHerc0841 volume is 113 keV / 1.2 m / binmean 2, and how its
       acquisition radix compares with the label meta's '4.681um_113keV_1.2m_binmean_2_PHerc_0841';
    3. the grower's own tifxyz_original meta: area_cm2 / area_vx2 gives the voxel size it was grown at;
    4. geometry: the original tifxyz (9.366 um frame) carried by the catalog matrix lies ON the published
       2.403 um tifxyz (point-to-plane distance), i.e. the 2.403 mesh is a transform of a 9 um-grown one.
WHAT WOULD FAIL IT: an original_volume_id other than 20250821151531; a grown voxel size near 2.403 um;
a median point-to-plane offset of several 2.403 um voxels (an independently traced surface).
"""
import json, os, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree

G = 'geom/day1-9um/gates'
S3 = os.path.join(G, 's3')

# ---- (a) -------------------------------------------------------------------------------------------
pe = json.load(open(os.path.join(G, 'prizeEligibility.json')))
fl = {e['scroll']: e['volume'] for e in pe['first-letters-2027']}
gp = {e['scroll']: e['volume'] for e in pe['grand-prize-2027']}
print('(a) prizeEligibility.json (fetched live, raw.githubusercontent.com ScrollPrize/villa main)')
print('    first-letters-2027: %d scrolls; grand-prize-2027: %d scrolls' % (len(fl), len(gp)))
for s, v in (('PHerc1203', '20250820131727'), ('PHerc0846A', '20250728152254')):
    ok = fl.get(s) == v
    print('    %-10s %s  first-letters: %s  grand-prize: %s  -> %s' % (
        s, v, fl.get(s, 'ABSENT'), gp.get(s, 'ABSENT'), 'STILL ELIGIBLE' if ok else 'NOT ELIGIBLE / CHANGED'))
print('    PHerc0841 first-letters: %s (control scroll)   PHerc1447 first-letters: %s grand-prize: %s' % (
    fl.get('PHerc0841', 'ABSENT'), fl.get('PHerc1447', 'ABSENT'), gp.get('PHerc1447', 'ABSENT')))

# ---- (b) -------------------------------------------------------------------------------------------
cat = json.load(open(os.path.join(G, 'catalog-live.json'), encoding='utf-8'))
S = cat['samples']
s841 = S['PHerc0841']
print('\n(b1) catalog (live metadata.json): PHerc0841 segments')
for k, g in s841['segments'].items():
    print('    %s  original_volume_id %s  derived_from %s  process %s  source %s' % (
        g['long_id'], g.get('original_volume_id'), g['creation'].get('derived_from'), g['creation'].get('process'),
        g['data'][0]['origins'][0].get('copy_info', {}).get('source_path')))

def radix(sample, scan_id):
    sc = S[sample]['scans'].get(scan_id)
    try:
        return sc['creation']['metadata']['metadata']['tomo']['acquisition']['scanRadix']
    except Exception:
        return '?'
print('\n(b2) catalog: volumes, their scans and acquisition radices')
for smp in ('PHerc0841', 'PHerc1203', 'PHerc0846A'):
    for vid, v in S[smp]['volumes'].items():
        p = v['properties']
        print('    %-10s %s  %-48s keV %5.1f  dist %6.1f mm  px %.3f um  scan %s  radix %s' % (
            smp, vid, v['long_id'], p.get('energy_keV') or 0, p.get('detector_distance_mm') or 0,
            p.get('pixel_size_um') or 0, v.get('scan_id'), radix(smp, v.get('scan_id'))))
label_meta_volume = '4.681um_113keV_1.2m_binmean_2_PHerc_0841'
hits = [smp for smp, s in S.items() for sid in s.get('scans', {}) if label_meta_volume in json.dumps(s['scans'][sid])]
print('    label meta volume string %r: exact matches in catalog scans: %s' % (label_meta_volume, hits or 'NONE'))
r841 = [radix('PHerc0841', sid) for sid in s841['scans']]
print('    PHerc0841 scan radices: %s' % r841)

print('\n(b3) grower meta (tifxyz_original/meta.json): grown voxel size = sqrt(area_cm2*1e8 / area_vx2)')
for seg in ('20260220213127-w00', '20260220214732-auto_grown_20260220144552896', '20260221022814-auto_grown_20260220174252405'):
    m = json.load(open(os.path.join(S3, seg, 'mesh/intermediate/tifxyz_original/meta.json')))
    vox = np.sqrt(m['area_cm2'] * 1e8 / m['area_vx2'])
    print('    %-46s area %.3f cm2 over %.0f vx2 -> %.3f um voxels; source %s; normal_grid_path %s' % (
        seg, m['area_cm2'], m['area_vx2'], vox, m.get('source'), (m.get('vc_gsfs_params') or {}).get('normal_grid_path')))

print('\n(b4) geometry: catalog matrix 20250821151531 -> 20260319124803 applied to tifxyz_original, distance to the')
print('     published 2.403 um tifxyz (point-to-plane from 8 nearest dense vertices), in 2.403 um voxels')
T = None
for e in s841['sample']['properties']['volume_transforms']:
    if e['from_volume_id'] == '20250821151531':
        for t in e['transforms']:
            if t['to_volume_id'] == '20260319124803':
                T = np.array(t['matrix'], np.float64)
assert T is not None and T.shape == (3, 4)
def load_xyz(d):
    X, Y, Z = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    ok = (X > 0) & (Y > 0) & (Z > 0)
    return np.stack([X[ok], Y[ok], Z[ok]], 1), X.shape
for seg, sid in (('20260220213127-w00', '20260220213127'), ('20260221022814-auto_grown_20260220174252405', '20260221022814'),
                 ('20260220214732-auto_grown_20260220144552896', '20260220214732')):
    P0, sh0 = load_xyz(os.path.join(S3, seg, 'mesh/intermediate/tifxyz_original'))
    P2, sh2 = load_xyz(os.path.join(S3, seg, 'mesh/%s-on-20260319124803-2.403um.tifxyz' % sid))
    Q = P0 @ T[:, :3].T + T[:, 3]                         # catalog matrices act on (x, y, z)
    tr = cKDTree(P2)
    d, k = tr.query(Q, k=8)
    nn = d[:, 0]
    off = np.empty(len(Q))
    for i in range(len(Q)):
        nb = P2[k[i]]
        c = nb.mean(0)
        _, _, vt = np.linalg.svd(nb - c)
        off[i] = abs((Q[i] - c) @ vt[2])
    # the null for "a different surface": the same points pushed 10 2.403-um voxels along the local normal
    print('    %-46s orig grid %s (%d pts) dense grid %s (%d pts)' % (seg, sh0, len(P0), sh2, len(P2)))
    print('      NN dist median %.2f p95 %.2f vox; point-to-plane median %.3f p95 %.3f max %.3f vox (%.2f um median)' % (
        np.median(nn), np.percentile(nn, 95), np.median(off), np.percentile(off, 95), off.max(), 2.403 * np.median(off)))
    print('      fraction of original vertices within 1 voxel of the 2.403 surface: %.4f' % (off < 1).mean())
    # the published chain is original -> normalized (re-meshed, 9.366 frame) -> transformed (2.403 frame);
    # test the last link vertex-exact: same grid, so compare cell by cell
    Pn, shn = load_xyz(os.path.join(S3, seg, 'mesh/intermediate/tifxyz_normalized'))
    Xn, Yn, Zn = (tifffile.imread(os.path.join(S3, seg, 'mesh/intermediate/tifxyz_normalized', k + '.tif')).astype(np.float64) for k in 'xyz')
    X2, Y2, Z2 = (tifffile.imread(os.path.join(S3, seg, 'mesh/%s-on-20260319124803-2.403um.tifxyz' % sid, k + '.tif')).astype(np.float64) for k in 'xyz')
    print('      normalized grid %s (%d pts); 2.403 grid %s' % (shn, len(Pn), X2.shape))
    Qn = Pn @ T[:, :3].T + T[:, 3]
    dn, kn = tr.query(Qn, k=8)
    offn = np.empty(len(Qn))
    for i in range(len(Qn)):
        nb = P2[kn[i]]; c = nb.mean(0); _, _, vt = np.linalg.svd(nb - c); offn[i] = abs((Qn[i] - c) @ vt[2])
    print('      normalized(9.366) -> catalog matrix -> 2.403 surface: NN median %.3f vox, point-to-plane median %.4f p95 %.4f vox' % (
        np.median(dn[:, 0]), np.median(offn), np.percentile(offn, 95)))
    # and original vs normalized, both in the 9.366 frame (the re-mesh step), in 9.366 um voxels
    trn = cKDTree(Pn)
    do, ko = trn.query(P0, k=8)
    offo = np.empty(len(P0))
    for i in range(len(P0)):
        nb = Pn[ko[i]]; c = nb.mean(0); _, _, vt = np.linalg.svd(nb - c); offo[i] = abs((P0[i] - c) @ vt[2])
    print('      original vs normalized (9.366 frame, re-mesh): point-to-plane median %.3f p95 %.3f 9.366-um voxels (%.2f um median)' % (
        np.median(offo), np.percentile(offo, 95), 9.366 * np.median(offo)))
