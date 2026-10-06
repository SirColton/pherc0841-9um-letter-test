"""ref-from-letters.py - turn a Reader A letters.json (the 252405 form) into the reference score.py reads.

  ref-from-letters.py --letters reference/letters.json --out REF.json

Reads letters.json alone: no map, no render, no 9 um anything.
  letters.json   box [row0, row1, col0, col1] in MAP pixels of its source map, HALF-OPEN, row 0 at top;
                 render_crop_map_px {rows: [r0, r1], cols: [c0, c1]} and map_pixel_um; the pool is the
                 crop's map rows per render row, (r1 - r0) / H, read from its "from" note (pool 4).
  REF.json       score.py's reference: letters [{id, letter, i, j, x_mm, y_mm, box_px, confidence, ...}]
                 in RENDER pixels, the same convention as the w00big reference and the pool keys:
                 integer i is the CENTRE of render pixel i. So a map edge coordinate x becomes
                 (x - r0) / pool - 0.5. letters.json's own centre_render_px (and its map_to_render note)
                 omit the 0.5, i.e. they are edge coordinates; the difference is half a render pixel.
                 box_px = [i_min, i_max, j_min, j_max] inclusive, as score.py reads it (it extends each
                 edge by half a pixel), so the half-open map box [row0, row1) becomes
                 i_min = (row0 - r0) / pool, i_max = (row1 - r0) / pool - 1 and covers exactly that extent.
SCORED: letters counted in N_ref (conf clear or probable) whose box centre lies inside the render crop
(half-open), recomputed here and refused if letters.json's own centre_in_render_crop disagrees.
Everything else goes in not_scored with the reason. The centre is the BOX centre (prereg v2), never the
ink centroid.
Writes once: refuses an existing --out. Lock it with score.py lock when the run's owner says so.
"""
import argparse
import json
import os
import re
import sys

COUNTED = ('clear', 'probable')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--letters', required=True)
    ap.add_argument('--out', required=True)
    A = ap.parse_args(argv)
    if os.path.exists(A.out):
        raise SystemExit('REFUSED: %s exists; a reference is written once' % A.out)
    d = json.load(open(A.letters, encoding='utf-8'))
    order = d.get('box_order', '')
    if not ('[row0, row1, col0, col1]' in order and 'half-open' in order):
        raise SystemExit('REFUSED: box_order %r is not "[row0, row1, col0, col1] ... half-open"' % order)
    crop = d['render_crop_map_px']
    (r0, r1), (c0, c1) = crop['rows'], crop['cols']
    m = re.search(r'pool (\d+), H (\d+), W (\d+)', crop.get('from', ''))
    if not m:
        raise SystemExit('REFUSED: render_crop_map_px.from names no "pool N, H h, W w"')
    pool, H, W = (int(x) for x in m.groups())
    if (r1 - r0, c1 - c0) != (pool * H, pool * W):
        raise SystemExit('REFUSED: crop %s x %s map px is not pool %d x (H %d, W %d)' % (r1 - r0, c1 - c0, pool, H, W))
    pitch = float(d['map_pixel_um']) * pool

    def ri(x):
        return (x - r0) / float(pool) - 0.5

    def rj(x):
        return (x - c0) / float(pool) - 0.5

    letters, not_scored = [], []
    for L in d['letters']:
        a, b, c, e = (float(v) for v in L['box'])
        if not (a < b and c < e):
            raise SystemExit('REFUSED: %s box %s is empty or out of order' % (L['id'], L['box']))
        cr, cc = (a + b) / 2.0, (c + e) / 2.0
        in_crop = r0 <= cr < r1 and c0 <= cc < c1
        if 'centre_in_render_crop' in L and bool(L['centre_in_render_crop']) != in_crop:
            raise SystemExit('REFUSED: %s centre (%g, %g) is %s the crop by this count but letters.json says %s' % (
                L['id'], cr, cc, 'inside' if in_crop else 'outside', L['centre_in_render_crop']))
        counted = L.get('conf') in COUNTED
        if 'counted_in_N_ref' in L and bool(L['counted_in_N_ref']) != counted:
            raise SystemExit('REFUSED: %s conf %r but counted_in_N_ref %s' % (L['id'], L.get('conf'), L['counted_in_N_ref']))
        i, j = ri(cr), rj(cc)
        rec = dict(id=L['id'], letter=L['char'], i=i, j=j, x_mm=round(j * pitch / 1000, 4), y_mm=round(i * pitch / 1000, 4),
                   box_px=[(a - r0) / pool, (b - r0) / pool - 1, (c - c0) / pool, (e - c0) / pool - 1],
                   confidence=L.get('conf'), line=L.get('line'), alt=L.get('alt', []), map_box=L['box'])
        if counted and in_crop:
            letters.append(rec)
        else:
            not_scored.append(dict(rec, why_out='; '.join(w for w in (
                None if counted else 'conf %r is not counted in N_ref' % L.get('conf'),
                None if in_crop else 'box centre outside the render crop') if w)))
    n_ref = d.get('summary', {}).get('N_ref_centre_in_render_crop')
    if n_ref is not None and n_ref != len(letters):
        raise SystemExit('REFUSED: %d letters scored here, letters.json summary says %d' % (len(letters), n_ref))
    out = dict(
        segment=d.get('source_map_note'), source_letters=os.path.abspath(A.letters), reader=d.get('reader'),
        coords='render pixels, H %d x W %d at %.3f um; integer i is the CENTRE of render pixel i; map edge x -> '
               '(x - %d) / %d - 0.5 rows, (x - %d) / %d - 0.5 cols. box_px = [i_min, i_max, j_min, j_max], '
               'inclusive, covering the half-open map box exactly' % (H, W, pitch, r0, pool, c0, pool),
        centre='box centre (prereg v2), not the ink centroid', pitch_um=pitch,
        letters=letters, not_scored=not_scored)
    with open(A.out, 'x', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('ref-from-letters: %d scored, %d not scored (%s) -> %s' % (
        len(letters), len(not_scored), ', '.join(x['id'] for x in not_scored), A.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
