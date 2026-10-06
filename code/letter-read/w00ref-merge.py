"""w00ref-merge.py - the nine w00 region readers' marks into one Reader A letters.json, in the 252405 form, so
ref-from-letters.py turns it into a score.py reference exactly as it did for 252405.

  w00ref-merge.py --marks WORKFLOW_RESULT.json --out reference-w00/letters.json

WORKFLOW_RESULT.json is the w00-reference-readers run's result: [{region: {k, core, window}, marks: {orientation,
letters: [{line, char, alt, conf, box, note}], notes}}]. Each box is [row0, row1, col0, col1] in render px of
w00-spmap-grid.npy, half-open. Refused, never repaired: a box that is empty or out of order, one outside the map, or one
whose centre lies outside its own reader's core (a duplicate in the making). The "map pixel" of this letters.json IS
the render pixel (pool 1), so ref-from-letters.py maps an edge x to x - 0.5, the integer-is-centre convention.
Writes once.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEOM = os.path.normpath(os.path.join(HERE, '..', '..'))
MAP = os.path.join(GEOM, 'student-9um', 'renders', 'w00-spmap-grid.npy')
H, W, PITCH = 4110, 4635, 9.612
COUNTED = ('clear', 'probable')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--marks', required=True)
    ap.add_argument('--out', required=True)
    A = ap.parse_args(argv)
    if os.path.exists(A.out):
        raise SystemExit('REFUSED: %s exists; a reference is written once' % A.out)
    runs = json.load(open(A.marks, encoding='utf-8'))
    runs = runs.get('result', runs) if isinstance(runs, dict) else runs
    if sorted(r['region']['k'] for r in runs) != list(range(9)):
        raise SystemExit('REFUSED: regions read %s, want 0-8' % sorted(r['region']['k'] for r in runs))
    letters, problems, orient = [], [], {}
    for run in sorted(runs, key=lambda r: r['region']['k']):
        k, core = run['region']['k'], run['region']['core']
        orient['r%d' % k] = run['marks']['orientation']
        for n, L in enumerate(run['marks']['letters'], 1):
            lid = 'R%d-%d' % (k, n)
            b = L['box']
            if not (len(b) == 4 and b[0] < b[1] and b[2] < b[3]):
                problems.append('%s: box %s empty or out of order' % (lid, b))
                continue
            if not (0 <= b[0] and b[1] <= H and 0 <= b[2] and b[3] <= W):
                problems.append('%s: box %s outside the map %d x %d' % (lid, b, H, W))
                continue
            cr, cc = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
            if not (core[0] <= cr < core[1] and core[2] <= cc < core[3]):
                problems.append('%s: box centre (%g, %g) outside region %d core %s' % (lid, cr, cc, k, core))
                continue
            letters.append(dict(id=lid, region=k, line=L['line'], char=L['char'], alt=L.get('alt') or [], conf=L['conf'],
                                box=[int(v) for v in b], note=L.get('note', ''), counted_in_N_ref=L['conf'] in COUNTED,
                                centre_map_px=[cr, cc], centre_in_render_crop=True,
                                size_mm_hw=[round((b[1] - b[0]) * PITCH / 1000, 2), round((b[3] - b[2]) * PITCH / 1000, 2)]))
    if problems:
        raise SystemExit('REFUSED, nothing written:\n  ' + '\n  '.join(problems))
    n_ref = sum(L['counted_in_N_ref'] for L in letters)
    out = dict(
        source_map='geom/student-9um/renders/w00-spmap-grid.npy',
        source_map_sha256=hashlib.sha256(open(MAP, 'rb').read()).hexdigest(),
        source_map_note="ScrollPrize's 2.4 um ink map of PHerc0841 w00 resampled to the 9.612 um render grid; w00big's "
                        'frame is rows [1900, 2895) cols [2500, 3745) of it',
        map_shape_rows_cols=[H, W], map_pixel_um=PITCH,
        render_crop_map_px={'rows': [0, H], 'cols': [0, W], 'from': 'w00-grid.json, pool 1, H %d, W %d' % (H, W)},
        map_to_render='render_i = row, render_j = col (the map IS the render grid)',
        box_order='[row0, row1, col0, col1] in map pixels of w00-spmap-grid.npy (= render px), half-open, row 0 at the top',
        reader='Reader A, nine fresh blind agents (w00-reference-readers), one per 3 x 3 region, each marking only letters '
               'centred in its core; each saw only views of w00-spmap-grid.npy drawn by w00ref-views.py',
        confidence_scale={'clear': 'identity certain from the shape alone; counted in N_ref',
                          'probable': 'identity read with one plausible alternative noted; counted in N_ref',
                          'possible': 'a letter-like shape whose identity is a guess; NOT counted in N_ref'},
        orientation_by_region=orient,
        summary=dict(N_ref=n_ref, N_ref_centre_in_render_crop=n_ref, N_possible_not_counted=len(letters) - n_ref,
                     N_all_marks=len(letters)),
        letters=letters)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, 'x', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('w00ref-merge: %d marks (%d counted, %d possible) from 9 regions -> %s' % (len(letters), n_ref, len(letters) - n_ref, A.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
