"""student-freeze-geom.py - the student branch's training cells, fixed from GEOMETRY before any render or training.

Part "freeze" of the twin-scan student (geom/student-9um/prereg-student.json). Writes, under
geom/student-9um/freeze/geom/, one boolean cell mask per training segment (w00, 552896) on its published
2.403 um tifxyz grid, the w00 validation strip, the distances behind every exclusion, and a log whose
asserts prove the training footprint stays clear of the held-out test text (252405).

WHAT IS READ OF 252405, and nothing else of it: the x/y/z of its published 2.403 um tifxyz mesh (S3 copy,
checked byte-equal to the mesh its render used) and the crop cells of its render (render/252405-grid.json).
No label, mask, map, stack or model output of 252405 is opened. The test frame is therefore the render
crop (every scored pixel lies inside it), not the supervised area, a superset chosen so that no label file
has to be read; the cost is training area, which the log states.

THE RULES (prereg-student.json fixes the same numbers; this script is their only implementation):
  3D   every voxel a training input can carry lies >= 50 um from the 252405 test-frame surface. An input
       spans layers 6..34 of the 41-layer render (depth centres -4..+4, 21-layer slab, 17-of-21 window),
       i.e. +-134.4 um along the normal, and trilinear sampling reaches sqrt(3) x 9.366 = 16.2 um further.
       So a cell's surface must sit >= 50 + 134.4 + 16.2 = 200.6 um from the test surface. Measured on
       both surfaces sampled 5x5 per quad (spacing <= L/4, L = 77.3 um, just above the largest edge of any mesh here, 77.23),
       which can overstate a distance by at most 2 x (L/4) x sqrt(2)/2 = 27.3 um, so the cut is 227.9 um.
       A cell owns the 4 quads around its vertex (mesh coords [i-1, i+1]), which covers both readings of
       which canvas block a vertex owns.
  2D   no training cell on 252405's sheet within 2 mm, measured on 252405's own grid (48.06 um per cell),
       of the test frame. A training vertex is on that sheet when it lies < 192.2 um (80 vox, the record's
       co-supervision radius, + 54.6 um because it is measured vertex to vertex) from 252405's surface; its flat position is its nearest 252405 vertex. Where
       that nearest vertex is on 252405's mesh boundary, the sheet may continue past it: a vertex whose
       normal offset there is < 300 um is read as the continuation, its flat distance the boundary's plus
       its tangential offset. Cut 2 mm + 2 cells (96 um, vertex-level slack). The 3x3 neighbourhood of an
       excluded vertex is excluded.
  VAL  w00 cells rows [380, 580) x cols [690, 740) (2.40 x 9.61 mm, inside w00big, clear of every 252405 cut) are the validation
       strip; any training cell (either segment) within 1.5 mm in 3D of it (+54.6 um vertex slack) is
       excluded. 3D Euclidean <= along-sheet distance, so this also removes 552896's copy of that text.
  DATA a cell whose 20x20 teacher block is all zero (the map did not predict there) is not trained on.
Then the render crop of each segment = bounding box of (allowed or validation) cells + 32 cells, clipped.

WHAT WOULD MAKE IT WORTHLESS, decided before running: the S3 252405 mesh differing from the render's mesh;
any allowed cell under the 3D cut or inside the 2D cut (asserted, the run stops); the val strip touching an
allowed cell; a teacher tif whose hash differs from the one the prereg names.
CPU only, local files. Run from <workdir>. Usage: student-freeze-geom.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

import runfile

OUT = 'geom/student-9um/freeze/geom'
os.makedirs(OUT, exist_ok=True)
REP = os.path.join(OUT, 'geom-run.txt')
runfile.claim(REP)
t0 = time.time()
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


VOX = 2.403                      # um per voxel of the tifxyz coordinates (2.403 um frame)
CELL_UM = 20 * 2.403             # 48.06 um per cell on the flattened grid (20 canvas px)
HALF_DEPTH_UM = 14 * 9.6         # layers 6..34 of 41 about centre 20
REACH_UM = np.sqrt(3) * 9.366    # trilinear support in the 9.366 um volume
RULE3D_UM = 50.0
LCAP_UM = 77.3                   # largest edge of any good quad below (measured, asserted)
SLACK_DENSE_UM = 2 * (LCAP_UM / 4) * np.sqrt(2) / 2
HALF_DIAG_UM = LCAP_UM * np.sqrt(2) / 2
T3D_UM = RULE3D_UM + HALF_DEPTH_UM + REACH_UM
CUT3D_UM = T3D_UM + SLACK_DENSE_UM
SHEET_UM = 80 * VOX
BOUNDARY_NU_UM = 300.0
RULE2D_UM = 2000.0
CUT2D_UM = RULE2D_UM + 2 * CELL_UM
VAL_ROWS, VAL_COLS = (380, 580), (690, 740)
VALBUF_UM = 1500.0
PAD_CELLS = 32

G = 'geom/day1-9um/gates/s3'
TRAIN = {
    'w00': (G + '/20260220213127-w00/mesh/20260220213127-on-20260319124803-2.403um.tifxyz',
            G + '/20260220213127-w00/ink-detection/PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif',
            'c980b60dc457037d0cfe8c654619c23f3fd673e2847d64aed09cbdb3e26f7bde'),
    '552896': (G + '/20260220214732-auto_grown_20260220144552896/mesh/20260220214732-on-20260319124803-2.403um.tifxyz',
               'geom/student-9um/freeze/teacher/PHerc0841-20260220214732-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif',
               '1c89e92daefbcfe21931f393f1cdc7b488e9a1d1c559e2b59eb2f730d34997aa'),
}
TEST_MESH = G + '/20260221022814-auto_grown_20260220174252405/mesh/20260221022814-on-20260319124803-2.403um.tifxyz'
TEST_RENDER_MESH = 'geom/day1-9um/render/src/mesh-2.403'
TEST_GRID_JSON = 'geom/day1-9um/render/252405-grid.json'


def load_mesh(d):
    X, Y, Z = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    return np.stack([X, Y, Z], -1), (X > 0) & (Y > 0) & (Z > 0)


def quads(P, ok):
    q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    e = np.max([np.linalg.norm(P[1:, :-1] - P[:-1, :-1], axis=-1), np.linalg.norm(P[:-1, 1:] - P[:-1, :-1], axis=-1),
                np.linalg.norm(P[1:, 1:] - P[1:, :-1], axis=-1), np.linalg.norm(P[1:, 1:] - P[:-1, 1:], axis=-1)], 0) * VOX
    return q, np.where(q, e, 0.0)


def dense(P, ii, jj, n=5):
    s = np.linspace(0.0, 1.0, n)
    A, B, C, D = P[ii, jj], P[ii + 1, jj], P[ii, jj + 1], P[ii + 1, jj + 1]
    out = np.empty((len(ii), n * n, 3))
    k = 0
    for a in s:
        for b in s:
            out[:, k] = (1 - a) * (1 - b) * A + a * (1 - b) * B + (1 - a) * b * C + a * b * D
            k += 1
    return out


def vertex_normals(P, ok):
    tr = np.zeros_like(P)
    tc = np.zeros_like(P)
    tr[1:-1] = P[2:] - P[:-2]
    tc[:, 1:-1] = P[:, 2:] - P[:, :-2]
    n = np.cross(tr, tc)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(nn > 0, n / np.maximum(nn, 1e-9), 0.0)


def nbhd3(mask):
    return binary_dilation(mask, structure=np.ones((3, 3), bool))


say('student-freeze-geom.py  %s' % time.strftime('%Y-%m-%dT%H:%M:%S%z'))
say('  cuts: 3D %.1f um (= 50 + %.1f half depth + %.1f trilinear reach + %.1f sampling slack); 2D %.0f um on 252405\'s grid '
    '(2 mm + 2 cells); same sheet < %.1f um (+%.1f vertex slack), boundary continuation normal offset < %.0f um; val buffer %.0f um + %.1f'
    % (CUT3D_UM, HALF_DEPTH_UM, REACH_UM, SLACK_DENSE_UM, CUT2D_UM, SHEET_UM, HALF_DIAG_UM, BOUNDARY_NU_UM, VALBUF_UM, HALF_DIAG_UM))

# ---- 252405: geometry only ---------------------------------------------------------------------------------------
same = all(sha(os.path.join(TEST_MESH, k + '.tif')) == sha(os.path.join(TEST_RENDER_MESH, k + '.tif')) for k in 'xyz')
say('  252405 S3 mesh x/y/z byte-equal to the mesh its render used (%s): %s' % (TEST_RENDER_MESH, same))
if not same:
    raise SystemExit('252405 mesh differs from the render mesh: stop')
grid = json.load(open(TEST_GRID_JSON, encoding='utf-8'))
r0, r1, c0, c1 = grid['crop_cells']
say('  252405 render crop cells rows [%d, %d) cols [%d, %d) (render %d x %d px at %.3f um), read from %s'
    % (r0, r1, c0, c1, grid['H'], grid['W'], grid['pitch_um_in_plane'], TEST_GRID_JSON))
PT, okT = load_mesh(TEST_MESH)
qT, eT = quads(PT, okT)
say('  252405 grid %s, valid vertices %d, good quads %d, largest quad edge %.1f um' % (PT.shape[:2], okT.sum(), qT.sum(), eT.max()))
assert eT.max() <= LCAP_UM, eT.max()
frame = np.zeros(qT.shape, bool)
frame[max(r0 - 1, 0):r1, max(c0 - 1, 0):c1] = True          # quads touching any crop vertex
fi, fj = np.nonzero(qT & frame)
DT = dense(PT, fi, fj).reshape(-1, 3)
treeD = cKDTree(DT)
say('  test frame surface: %d quads sampled 5x5 -> %d points' % (len(fi), len(DT)))
iv, jv = np.nonzero(okT)
treeV = cKDTree(PT[iv, jv])
nT = vertex_normals(PT, okT)
bnd = nbhd3(~okT) | np.pad(np.zeros((PT.shape[0] - 2, PT.shape[1] - 2), bool), 1, constant_values=True)
bnd &= okT


def flat_to_frame(i, j):
    dr = np.maximum(np.maximum(r0 - i, 0), i - (r1 - 1))
    dc = np.maximum(np.maximum(c0 - j, 0), j - (c1 - 1))
    return CELL_UM * np.hypot(dr, dc)


# ---- the validation strip on w00 -------------------------------------------------------------------------------
Pw, okw = load_mesh(TRAIN['w00'][0])
V = np.zeros(okw.shape, bool)
V[VAL_ROWS[0]:VAL_ROWS[1], VAL_COLS[0]:VAL_COLS[1]] = True
V &= okw
Vd = nbhd3(V) & okw
treeVal = cKDTree(Pw[Vd])
say('  validation strip: w00 rows [%d, %d) cols [%d, %d): %d valid cells (%.3f cm2), inside w00big rows [380, 580) cols [500, 750)'
    % (VAL_ROWS + VAL_COLS + (V.sum(), V.sum() * CELL_UM ** 2 / 1e8)))

summary = {'cuts_um': {'3D': CUT3D_UM, '3D_rule': RULE3D_UM, 'half_depth': HALF_DEPTH_UM, 'trilinear_reach': REACH_UM,
                       'sampling_slack': SLACK_DENSE_UM, '2D': CUT2D_UM, 'same_sheet': SHEET_UM,
                       'boundary_normal_offset': BOUNDARY_NU_UM, 'val_buffer': VALBUF_UM + HALF_DIAG_UM},
           'test_frame': {'crop_cells': [r0, r1, c0, c1], 'quads': int(len(fi)), 'points': int(len(DT)),
                          'mesh_equal_to_render_mesh': bool(same)},
           'val_strip': {'segment': 'w00', 'rows': list(VAL_ROWS), 'cols': list(VAL_COLS), 'cells': int(V.sum())},
           'segments': {}, 'files': {}}

for name, (mdir, tpath, tsha) in TRAIN.items():
    say()
    say('== %s' % name)
    got = sha(tpath)
    say('  teacher %s sha256 %s (%s)' % (os.path.basename(tpath), got, 'matches' if got == tsha else 'MISMATCH'))
    if got != tsha:
        raise SystemExit('teacher hash mismatch: stop')
    P, ok = (Pw, okw) if name == 'w00' else load_mesh(mdir)
    H, W = ok.shape
    q, e = quads(P, ok)
    say('  grid %s, valid vertices %d (%.2f cm2), good quads %d, largest quad edge %.1f um' % ((H, W), ok.sum(), ok.sum() * CELL_UM ** 2 / 1e8, q.sum(), e.max()))
    assert e.max() <= LCAP_UM, e.max()
    tif = tifffile.imread(tpath)
    assert tif.shape == (20 * H, 20 * W), (tif.shape, H, W)
    tvalid = tif.reshape(H, 20, W, 20).max((1, 3)) > 0
    del tif

    # 3D: vertex distances to the test frame, refined on 5x5-sampled quads where they come near the cut
    vi, vj = np.nonzero(ok)
    dv = np.full(ok.shape, np.inf)
    dv[vi, vj] = treeD.query(P[vi, vj], workers=-1)[0] * VOX
    corner = np.minimum(np.minimum(dv[:-1, :-1], dv[1:, :-1]), np.minimum(dv[:-1, 1:], dv[1:, 1:]))
    cand = q & (corner < CUT3D_UM + HALF_DIAG_UM + 1.0)
    dq = np.where(q, corner - HALF_DIAG_UM, np.inf)
    ci, cj = np.nonzero(cand)
    if len(ci):
        pts = dense(P, ci, cj)
        d = treeD.query(pts.reshape(-1, 3), workers=-1)[0].reshape(len(ci), -1) * VOX
        dq[ci, cj] = d.min(1)
    dcell = np.full(ok.shape, np.inf)
    for a in (0, 1):
        for b in (0, 1):
            dcell[a:H - 1 + a, b:W - 1 + b] = np.minimum(dcell[a:H - 1 + a, b:W - 1 + b], dq)
    lone = ok & ~np.isfinite(dcell)
    dcell[lone] = dv[lone] - HALF_DIAG_UM
    ex3 = ok & (dcell < CUT3D_UM)
    say('  3D: %d quads refined; vertex min %.1f um; cell min %.1f um; cells under %.1f um: %d (%.2f cm2)'
        % (len(ci), dv[ok].min(), dcell[ok].min(), CUT3D_UM, ex3.sum(), ex3.sum() * CELL_UM ** 2 / 1e8))

    # 2D: on 252405's sheet (overlap) or its continuation past its mesh boundary, within 2 mm of the frame
    dn, kn = treeV.query(P[vi, vj], workers=-1)
    dn = dn * VOX
    ti, tj = iv[kn], jv[kn]
    flat = flat_to_frame(ti, tj)
    v = (P[vi, vj] - PT[ti, tj]) * VOX
    nu = np.abs(np.einsum('ij,ij->i', v, nT[ti, tj]))
    tau = np.sqrt(np.maximum((v * v).sum(1) - nu * nu, 0))
    on_sheet = dn < SHEET_UM + HALF_DIAG_UM
    cont = ~on_sheet & bnd[ti, tj] & (nu < BOUNDARY_NU_UM)
    dflat = np.where(on_sheet, flat, np.where(cont, flat + tau, np.inf))
    hit = dflat < CUT2D_UM
    hitg = np.zeros(ok.shape, bool)
    hitg[vi[hit], vj[hit]] = True
    ex2 = nbhd3(hitg) & ok
    D2 = np.full(ok.shape, np.inf)
    D2[vi, vj] = dflat
    say('  2D: on 252405\'s sheet %d vertices (%d within the cut), boundary continuation %d (%d within the cut); '
        'cells excluded %d (%.2f cm2), of them not already under the 3D cut %d'
        % (on_sheet.sum(), (on_sheet & hit).sum(), cont.sum(), (cont & hit).sum(), ex2.sum(), ex2.sum() * CELL_UM ** 2 / 1e8, (ex2 & ~ex3).sum()))
    amb = ~on_sheet & bnd[ti, tj] & (nu >= BOUNDARY_NU_UM) & (nu < 2 * BOUNDARY_NU_UM) & (flat + tau < CUT2D_UM)
    say('     beside (not excluded): boundary-nearest vertices with normal offset %.0f-%.0f um and flat+tangential < cut: %d'
        % (BOUNDARY_NU_UM, 2 * BOUNDARY_NU_UM, amb.sum()))

    # validation strip and its buffer
    dval = np.full(ok.shape, np.inf)
    dval[vi, vj] = treeVal.query(P[vi, vj], workers=-1)[0] * VOX
    dval3 = dval.copy()
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            sh = np.full(ok.shape, np.inf)
            sh[max(a, 0):H + min(a, 0), max(b, 0):W + min(b, 0)] = dval[max(-a, 0):H + min(-a, 0), max(-b, 0):W + min(-b, 0)]
            dval3 = np.minimum(dval3, sh)
    exv = ok & (dval3 < VALBUF_UM + HALF_DIAG_UM)
    isval = V if name == 'w00' else np.zeros(ok.shape, bool)
    say('  val: buffer excludes %d cells (%.2f cm2)%s' % (exv.sum(), exv.sum() * CELL_UM ** 2 / 1e8,
                                                           ', strip itself %d cells' % isval.sum() if name == 'w00' else ''))

    noteach = ok & ~tvalid
    allowed = ok & tvalid & ~ex3 & ~ex2 & ~exv & ~isval
    reason = (np.uint8(1) * ~ok + np.uint8(2) * noteach + np.uint8(4) * ex3 + np.uint8(8) * ex2 + np.uint8(16) * isval + np.uint8(32) * exv).astype(np.uint8)

    # ASSERTS (the run stops on any)
    assert not (allowed & (dcell < CUT3D_UM)).any(), '3D cut violated'
    assert not (allowed & nbhd3(hitg)).any(), '2D cut violated'
    if name == 'w00':
        assert not (allowed & nbhd3(V)).any(), 'val strip touches an allowed cell'
    assert not (allowed & (dval3 < VALBUF_UM + HALF_DIAG_UM)).any(), 'val buffer violated'
    amin3 = float(dcell[allowed].min())
    footprint = amin3 - SLACK_DENSE_UM - HALF_DEPTH_UM - REACH_UM
    say('  ASSERT 3D: min cell distance over allowed cells %.1f um >= %.1f -> min footprint-to-frame distance >= %.1f um (rule 50): PASS'
        % (amin3, CUT3D_UM, footprint))
    a2 = float(D2[allowed].min()) if np.isfinite(D2[allowed]).any() else float('inf')
    say('  ASSERT 2D: min flat distance over allowed vertices that are on or continue 252405\'s sheet %s (cut %.0f): PASS'
        % ('none on its sheet' if not np.isfinite(a2) else '%.0f um' % a2, CUT2D_UM))
    say('  ASSERT VAL: min 3D distance of allowed cells to the strip %.0f um (>= %.1f): PASS' % (float(dval3[allowed].min()), VALBUF_UM + HALF_DIAG_UM))

    # patch feasibility, cell approximation of a 128 px window (25.6 cells -> 26)
    from scipy.ndimage import uniform_filter
    k = 26
    bad = (ok & ~allowed).astype(np.float32)
    good = allowed.astype(np.float32)
    fb = uniform_filter(bad, k, mode='constant') * k * k
    fg = uniform_filter(good, k, mode='constant') * k * k
    feas = (fb < 0.5) & (fg >= 0.5 * k * k)
    cover = binary_dilation(feas, structure=np.ones((k, k), bool)) & allowed
    say('  allowed cells %d (%.2f cm2 of %.2f valid); patch-feasible window centres %d; allowed cells inside some feasible window %d (%.2f cm2)'
        % (allowed.sum(), allowed.sum() * CELL_UM ** 2 / 1e8, ok.sum() * CELL_UM ** 2 / 1e8, feas.sum(), cover.sum(), cover.sum() * CELL_UM ** 2 / 1e8))
    keep = allowed | isval
    rr, cc = np.nonzero(keep)
    crop = [int(max(rr.min() - PAD_CELLS, 0)), int(min(rr.max() + 1 + PAD_CELLS, H)), int(max(cc.min() - PAD_CELLS, 0)), int(min(cc.max() + 1 + PAD_CELLS, W))]
    say('  render crop (allowed or val, + %d cells): rows [%d, %d) cols [%d, %d) -> %d x %d render px (day1-render-9um.py: 5 x (cells - 1))'
        % ((PAD_CELLS,) + tuple(crop) + (5 * (crop[1] - crop[0] - 1), 5 * (crop[3] - crop[2] - 1))))

    files = {}
    for tag, arr in (('allowed', allowed), ('reason', reason), ('dcell3d-um', dcell.astype(np.float32)), ('dflat2d-um', D2.astype(np.float32)),
                     ('dval-um', dval3.astype(np.float32))):
        p = os.path.join(OUT, '%s-%s.npy' % (name, tag))
        np.save(p, arr)
        files[p.replace('\\', '/')] = sha(p)
    if name == 'w00':
        p = os.path.join(OUT, 'w00-valstrip.npy')
        np.save(p, V)
        files[p.replace('\\', '/')] = sha(p)
    for p, h in files.items():
        say('  wrote %s sha256 %s' % (p, h))
    summary['files'].update(files)
    summary['segments'][name] = {
        'grid': [H, W], 'valid_cells': int(ok.sum()), 'teacher_sha256': got, 'teacher_empty_cells': int(noteach.sum()),
        'excluded_3d': int(ex3.sum()), 'excluded_2d': int(ex2.sum()), 'excluded_2d_only': int((ex2 & ~ex3).sum()),
        'val_strip': int(isval.sum()), 'excluded_val_buffer': int(exv.sum()),
        'allowed_cells': int(allowed.sum()), 'allowed_cm2': round(float(allowed.sum() * CELL_UM ** 2 / 1e8), 3),
        'patch_covered_cm2': round(float(cover.sum() * CELL_UM ** 2 / 1e8), 3),
        'min_vertex_dist_to_frame_um': round(float(dv[ok].min()), 1), 'min_allowed_cell_dist_um': round(amin3, 1),
        'min_footprint_dist_um': round(footprint, 1), 'render_crop_cells': crop,
        'on_sheet_vertices': int(on_sheet.sum()), 'continuation_vertices': int(cont.sum())}

say()
say('  %.0f s' % (time.time() - t0))
jp = os.path.join(OUT, 'geom.json')
json.dump(summary, open(jp, 'w', encoding='utf-8'), indent=1)
say('  wrote %s sha256 %s' % (jp.replace('\\', '/'), sha(jp)))
runfile.done(REP, '\n'.join(lines) + '\n')
