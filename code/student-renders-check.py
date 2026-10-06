"""student-renders-check.py - part renders of geom/student-9um/prereg-student.json: the w00big equality
check, and renders.json describing every grid the part wrote.

THE CHECK (prereg 2_training_regions.renders.w00): on the w00big window (render rows [1900, 2895), cols
[2500, 3745)) the full-w00 render must equal day 1's w00big stack wherever both masks are set, max |diff| 0.
Compared against BOTH copies the record holds: geom/day1-9um/render/w00big-republished/w00big-stack.npy
(sha256 pinned in the prereg) and geom/l0-1203-render/audit/build-deciding-test/ours-stack.npy (the stack
the republished one reproduced with 0 differing voxels). Reported: masks equal or not, voxels compared,
voxels differing, max |diff| overall, per layer, and split into the window's outer 5 px frame and its
interior - decided before looking: the w00big stacks were rendered from a 200 x 250 cell crop of the mesh,
where np.gradient takes one-sided differences at the crop's border cells, so the normals of the border
cells (render px within 5 of the window edge) are not the full mesh's central differences; layer 20 (the
mesh itself, no normal) must be equal everywhere, and the interior must be equal on every layer.
Also by index: the window of each canvas grid against w00big's own grids (same canvas, same rule).

RENDERS.JSON: per segment the grid origin (crop cells, canvas origin px), the px -> cell and px -> canvas
rules, shape, pitch, stored order, sha256 of every file, mask area in cm2, teacher-valid and labelled
areas, gate 1, the chunk count, and the disk used.

Never opens a 252405 file. CPU only. Usage: student-renders-check.py. Run from <workdir>.
"""
import glob
import hashlib
import json
import os
import time

import numpy as np

import runfile

OUTD = 'geom/student-9um/renders'
OUT = os.path.join(OUTD, 'renders-check-run.txt')
runfile.claim(OUT)
lines = []
CM2 = (2.403 * 4 * 1e-4) ** 2


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


say('student-renders-check.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
pre = sha256('geom/student-9um/prereg-student.json')
say('  prereg-student.json sha256 %s (%s)' % (pre, 'FROZEN' if pre == 'f3ebf0abc3f93a6772ce31f6915d7f5aef2ea4afc72badeb11cbb735216d3710' else 'DIFFERS'))

# ---- the w00big equality check --------------------------------------------------------------------------
WR, WC = (1900, 2895), (2500, 3745)
st = np.load(os.path.join(OUTD, 'w00-stack.npy'), mmap_mode='r')
mk = np.load(os.path.join(OUTD, 'w00-mask.npy'))
sub = np.asarray(st[:, WR[0]:WR[1], WC[0]:WC[1]])
msub = mk[WR[0]:WR[1], WC[0]:WC[1]]
F = 5
frame = np.zeros(msub.shape, bool)
frame[:F] = frame[-F:] = True
frame[:, :F] = frame[:, -F:] = True
check = {}
for label, sp, mp, pin in (('w00big-republished', 'geom/day1-9um/render/w00big-republished/w00big-stack.npy', 'geom/day1-9um/render/w00big-republished/w00big-mask.npy',
                            '8187275cb476c5352339b94819234529cbe683071b74ced1ba722bb280dbcd59'),
                           ('ours-stack', 'geom/l0-1203-render/audit/build-deciding-test/ours-stack.npy', 'geom/l0-1203-render/audit/build-deciding-test/ours-mask.npy', None)):
    s = sha256(sp)
    ref = np.load(sp)
    rm = np.load(mp)
    both = msub & rm
    d = np.abs(ref.astype(np.int16) - sub.astype(np.int16))
    d[:, ~both] = 0
    dl = [int(d[k].max()) for k in range(d.shape[0])]
    fi, it = both & frame, both & ~frame
    r = {'ref': sp, 'ref_sha256': s, 'ref_sha256_pinned': pin, 'pinned_ok': (s == pin) if pin else None,
         'shape_ref': list(ref.shape), 'shape_ours_window': list(sub.shape), 'mask_equal': bool((rm == msub).all()),
         'mask_px_ours': int(msub.sum()), 'mask_px_ref': int(rm.sum()), 'both_px': int(both.sum()),
         'voxels_compared': int(both.sum()) * d.shape[0], 'voxels_differing': int((d > 0).sum()), 'max_abs_diff': int(d.max()),
         'max_abs_diff_per_layer': dl, 'centre_layer_20_max': dl[20],
         'interior_px': int(it.sum()), 'interior_voxels_differing': int((d[:, it] > 0).sum()), 'interior_max_abs_diff': int(d[:, it].max()) if it.any() else 0,
         'frame_px': int(fi.sum()), 'frame_voxels_differing': int((d[:, fi] > 0).sum()), 'frame_max_abs_diff': int(d[:, fi].max()) if fi.any() else 0}
    if r['voxels_differing']:
        k_, y_, x_ = np.nonzero(d)
        r['differing_rows_range'] = [int(y_.min()), int(y_.max())]
        r['differing_cols_range'] = [int(x_.min()), int(x_.max())]
        r['differing_min_dist_to_window_edge_px_max'] = int(np.max(np.minimum.reduce([y_, d.shape[1] - 1 - y_, x_, d.shape[2] - 1 - x_])))
        r['differing_layers'] = sorted(set(int(v) for v in k_))
    check[label] = r
    say('')
    say('  W00BIG CHECK vs %s (sha256 %s%s):' % (sp, s[:16], '' if pin is None else (', pinned ' + ('OK' if s == pin else 'MISMATCH'))))
    say('    window rows %d-%d cols %d-%d of the full w00 render; shapes ours %s ref %s; masks equal %s (ours %d px, ref %d px, both %d px)'
        % (WR[0], WR[1], WC[0], WC[1], sub.shape, ref.shape, r['mask_equal'], r['mask_px_ours'], r['mask_px_ref'], r['both_px']))
    say('    voxels compared %d, differing %d, max |diff| %d; centre layer 20 max %d' % (r['voxels_compared'], r['voxels_differing'], r['max_abs_diff'], dl[20]))
    say('    interior (more than %d px from the window edge): %d px, voxels differing %d, max |diff| %d' % (F, r['interior_px'], r['interior_voxels_differing'], r['interior_max_abs_diff']))
    say('    outer %d px frame: %d px, voxels differing %d, max |diff| %d' % (F, r['frame_px'], r['frame_voxels_differing'], r['frame_max_abs_diff']))
    say('    per-layer max |diff|: %s' % ' '.join(str(v) for v in dl))
    if r['voxels_differing']:
        say('    differing voxels lie at window rows %s cols %s; the farthest from the window edge is %d px in; layers %s'
            % (r['differing_rows_range'], r['differing_cols_range'], r['differing_min_dist_to_window_edge_px_max'], r['differing_layers']))
    del ref, d
grid_check = {}
for nm in ('spmap', 'inklabels', 'supervision'):
    g = np.load(os.path.join(OUTD, 'w00-%s-grid.npy' % nm))[WR[0]:WR[1], WC[0]:WC[1]]
    o = np.load('geom/day1-9um/render/w00big-republished/w00big-%s-grid.npy' % nm)
    dd = np.abs(g - o)
    grid_check[nm] = {'max_abs_diff': float(dd.max()), 'px_differing': int((dd > 0).sum()), 'n': int(dd.size)}
    say('  grid %-12s window vs w00big-%s-grid.npy: max |diff| %.3g, px differing %d of %d' % (nm, nm, dd.max(), (dd > 0).sum(), dd.size))
g = np.load(os.path.join(OUTD, 'w00-validation-grid.npy'))[WR[0]:WR[1], WC[0]:WC[1]]
o = np.load('geom/day1-9um/render/w00big-republished/w00big-validation-grid.npy')
dd = np.abs(g - o)
grid_check['validation (student-renders-labels.py)'] = {'max_abs_diff': float(dd.max()), 'px_differing': int((dd > 0).sum()), 'n': int(dd.size)}
say('  grid validation   window vs w00big-validation-grid.npy: max |diff| %.3g, px differing %d of %d' % (dd.max(), (dd > 0).sum(), dd.size))
del st, sub

# ---- every S3-labelled canvas px of w00 inside the render (552896's is in its labels run) ---------------
import zarr
cover = {}
for nm in ('inklabels', 'supervision'):
    cv = zarr.open_array('geom/day1-9um/render/labelcheck/w00-20260918/%s.zarr/0' % nm, mode='r')[:]
    cv = cv.reshape(cv.shape[-2:]) if cv.ndim > 2 else cv
    pos = cv > 0
    H_, W_ = mk.shape
    ins = pos[:H_ * 4, :W_ * 4]
    cover[nm] = {'canvas_px_gt0': int(pos.sum()), 'in_render_extent': int(ins.sum()),
                 'in_render_mask': int(ins.reshape(H_, 4, W_, 4).transpose(0, 2, 1, 3)[mk].sum())}
    say('  w00 S3 20260918 %-12s canvas px > 0: %d; inside the render extent %d; inside the render mask %d'
        % (nm, cover[nm]['canvas_px_gt0'], cover[nm]['in_render_extent'], cover[nm]['in_render_mask']))
    del cv, pos, ins

# ---- renders.json ----------------------------------------------------------------------------------------
geomj = json.load(open('geom/student-9um/freeze/geom/geom.json'))
segs = {}
for seg in ('w00', '552896'):
    gj = json.load(open(os.path.join(OUTD, '%s-grid.json' % seg)))
    lj = json.load(open(os.path.join(OUTD, '%s-labels.json' % seg)))
    g1 = json.load(open(os.path.join(OUTD, '%s-gate1.json' % seg)))
    mask = np.load(os.path.join(OUTD, '%s-mask.npy' % seg))
    H, W = mask.shape
    sp = np.load(os.path.join(OUTD, '%s-spmap-grid.npy' % seg))
    # teacher-valid cell = its 20 x 20 teacher block not all zero (prereg 1_teacher.teacher_valid), by the px -> cell rule
    r0, c0 = gj['crop_cells'][0], gj['crop_cells'][2]
    cell_r = r0 + np.arange(H) // 5
    cell_c = c0 + np.arange(W) // 5
    allowed = np.load('geom/student-9um/freeze/geom/%s-allowed.npy' % seg)
    allowed_px = allowed[np.ix_(cell_r, cell_c)] & mask
    files = sorted(glob.glob(os.path.join(OUTD, '%s-*' % seg)))
    fs = {os.path.basename(f): {'bytes': os.path.getsize(f), 'sha256': sha256(f)} for f in files if os.path.isfile(f)}
    run = open(os.path.join(OUTD, '%s-run.txt' % seg), encoding='utf-8').read()
    chunk_line = [ln.strip() for ln in run.splitlines() if 'chunks touched' in ln or ln.strip().startswith('chunks:')]
    lab = np.load(os.path.join(OUTD, '%s-inklabels_v2-grid.npy' % seg))
    sup = np.load(os.path.join(OUTD, '%s-supervision_mask_v2-grid.npy' % seg))
    s3l = np.load(os.path.join(OUTD, '%s-inklabels-grid.npy' % seg))
    s3s = np.load(os.path.join(OUTD, '%s-supervision-grid.npy' % seg))
    a = {
        'mask_px': int(mask.sum()), 'mask_cm2': float(mask.sum() * CM2),
        'teacher_nonzero_px_in_mask': int((mask & (sp > 0)).sum()), 'teacher_nonzero_cm2': float((mask & (sp > 0)).sum() * CM2),
        'allowed_cell_px_in_mask': int(allowed_px.sum()), 'allowed_cell_cm2': float(allowed_px.sum() * CM2),
        'hf_v2_pos_cm2': float((mask & (lab >= 0.75)).sum() * CM2), 'hf_v2_neg_sup_cm2': float((mask & (lab <= 0) & (sup >= 0.75)).sum() * CM2),
        'hf_v2_supervised_or_ink_cm2': float((mask & ((sup >= 0.75) | (lab >= 0.75))).sum() * CM2),
        's3_20260918_pos_cm2': float((mask & (s3l >= 0.75)).sum() * CM2), 's3_20260918_neg_sup_cm2': float((mask & (s3l <= 0) & (s3s >= 0.75)).sum() * CM2),
        's3_20260918_supervised_or_ink_cm2': float((mask & ((s3s >= 0.75) | (s3l >= 0.75))).sum() * CM2)}
    segs[seg] = {
        'mesh': gj['mesh'], 'crop_cells_r0_r1_c0_c1': gj['crop_cells'], 'canvas_origin_px': gj['canvas_origin_px'],
        'H': H, 'W': W, 'NL': gj['NL'], 'pitch_um_in_plane': gj['pitch_um_in_plane'], 'pitch_um_depth': gj['pitch_um_depth'],
        'depth_step_voxels': gj['depth_step_voxels'], 'centre_layer': gj['centre_layer'], 'stored_along': gj['stored_along'], 'flipped': gj['flipped'],
        'orient': gj['orient'], 'volume': gj['volume'], 'affine': gj['affine'], 'stack_sha1_12': gj['stack_sha1'],
        'px_to_mesh': 'render px (i, j) samples the mesh at row r0 + (4 i + 1.5) / 20, col c0 + (4 j + 1.5) / 20 (bilinear), cell (r0 + i // 5, c0 + j // 5)',
        'px_to_canvas': gj['render_px_to_canvas'] + ', R0 = 20 r0, C0 = 20 c0 (2.403 um canvas, 20 px per cell)',
        'depth': 'layer k (0..40) = mesh point + (k - 20) x 9.6 um along %s the unit mesh normal (cross(d/drow, d/dcol) of the coarse z, y, x mesh), trilinear' % ('MINUS' if gj['flipped'] else 'PLUS'),
        'grids': {'spmap': 'teacher, ScrollPrize 2.4 um map by index (student-render-9um.py --canvas)',
                  'inklabels / supervision': 'S3 20260918 by index (%s)' % ('render command' if seg == 'w00' else 'student-renders-labels.py'),
                  'validation': 'S3 20260918 by index (student-renders-labels.py)',
                  'inklabels_v2 / supervision_mask_v2 / validation_mask_v2': "HF v2 by day1-labels-hf.py's 3D-nearest convention (student-renders-labels.py)",
                  'hf-nn-dist': 'grid point -> HF tifxyz nearest distance, 2.403 um voxels'},
        'areas': a, 'gate1': g1, 'chunks': chunk_line, 'hf_nn_dist': lj['hf_nn_dist_render_mask'], 'files': fs,
        'disk_bytes': int(sum(v['bytes'] for v in fs.values()))}
    say('')
    say('  %s: grid %d x %d x %d layers, crop cells %s, canvas origin %s, stored along %s; stack sha1 %s'
        % (seg, H, W, gj['NL'], gj['crop_cells'], gj['canvas_origin_px'], gj['stored_along'], gj['stack_sha1']))
    say('    render mask %d px = %.3f cm2; teacher non-zero %.3f cm2; allowed-cell px %.3f cm2' % (a['mask_px'], a['mask_cm2'], a['teacher_nonzero_cm2'], a['allowed_cell_cm2']))
    say('    HF v2: pos %.3f cm2, neg(sup) %.3f cm2, supervised-or-ink %.3f cm2 | S3 20260918: pos %.3f, neg(sup) %.3f, supervised-or-ink %.3f cm2'
        % (a['hf_v2_pos_cm2'], a['hf_v2_neg_sup_cm2'], a['hf_v2_supervised_or_ink_cm2'], a['s3_20260918_pos_cm2'], a['s3_20260918_neg_sup_cm2'], a['s3_20260918_supervised_or_ink_cm2']))
    say('    gate 1 (model-free, direct render vs spmap): %s' % json.dumps(g1))
    say('    %s' % ' | '.join(chunk_line))
    say('    files %d, %.3f GB on disk' % (len(fs), segs[seg]['disk_bytes'] / 1e9))
    del mask, sp, lab, sup, s3l, s3s, allowed_px
total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUTD, '**', '*'), recursive=True) if os.path.isfile(f))
say('')
say('  renders/ total on disk %.3f GB' % (total / 1e9))
json.dump({'part': 'renders', 'prereg_sha256': pre, 'script_render': 'student-render-9um.py (byte copy of day1-render-9um.py, sha256 155f35f1...)',
           'w00big_check': check, 'w00big_grid_check': grid_check, 'w00_s3_label_coverage': cover, 'segments': segs, 'renders_dir_bytes': total,
           'written': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(os.path.join(OUTD, 'renders.json'), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
