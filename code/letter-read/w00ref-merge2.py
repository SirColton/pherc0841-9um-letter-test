"""w00ref-merge2.py - the second w00 reference pass: two independent readers per region, kept by agreement, into one
Reader A letters.json in the 252405 form (map px of the native 2.4 um w00 tif, pool 4 to the render grid), so
ref-from-letters.py converts it exactly as it did 252405. Rules are a reviewer's ruling of 2026-10-06:

  AGREEMENT  a letter is kept only when BOTH readers of its region mark it: identities agree (the same letter, or one
             reader's letter among the other's listed alternatives) and their boxes overlap with IoU >= 0.3; pairs are
             taken greedily by IoU, one to one. Its box is the two boxes' mean; its letter is the more confident
             reader's (reader a on a tie); alt is everything else either reader named; it is COUNTED (clear/probable)
             when either reader counted it, since both agree it is present.
  DEDUPE     a letter kept by two neighbouring regions (IoU >= 0.3, identities agreeing) is kept once: the copy whose
             centre lies in its own region's core, else the more confident, else the lower region.
  REFUSED    malformed input (a region without both readers, a box empty or out of order) writes nothing; a mark off
             the map or centred outside its reader's marking zone is dropped and listed, never repaired.

  w00ref-merge2.py --marks WORKFLOW_RESULT.json --out reference-w00/letters-pass2.json
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('score', os.path.join(HERE, 'score.py'))
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)
RH, RW, POOL, PX_UM, IOU = 4110, 4635, 4, 2.403, 0.3
H, W = RH * POOL, RW * POOL
RANK = {'clear': 2, 'probable': 1, 'possible': 0}


def iou(a, b):
    r = max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[2], b[2]))
    u = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - r
    return r / float(u) if u else 0.0


def names(L):
    main = score.norm_letter(L['char'])
    alts = {score.norm_letter(x) for x in (L.get('alt') or [])} - {None}
    return main, alts


def agree(a, b):
    ma, aa = names(a)
    mb, ab = names(b)
    return ma is not None and mb is not None and (ma == mb or ma in ab or mb in aa)


def inside(c, box):
    return box[0] <= c[0] < box[1] and box[2] <= c[1] < box[3]


def merge(runs):
    byreg, dropped = {}, []
    for run in runs:
        k, rd = run['region']['k'], run['reader']
        for n, L in enumerate(run['marks']['letters'], 1):
            b = L['box']
            tag = 'r%d%s-%d' % (k, rd, n)
            if not (len(b) == 4 and b[0] < b[1] and b[2] < b[3]):
                raise SystemExit('REFUSED: %s box %s empty or out of order' % (tag, b))
            c = ((b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0)
            if not (0 <= b[0] and b[1] <= H and 0 <= b[2] and b[3] <= W):
                dropped.append('%s: box %s off the map' % (tag, b))
                continue
            if not inside(c, run['region']['zone']):
                dropped.append('%s: centre (%g, %g) outside region %d marking zone' % (tag, c[0], c[1], k))
                continue
            byreg.setdefault(k, {'region': run['region'], 'a': [], 'b': []})[rd].append(dict(L, tag=tag, box=[int(v) for v in b]))
    missing = [(k, x) for k in range(9) for x in 'ab' if not any(r['region']['k'] == k and r['reader'] == x for r in runs)]
    if missing:
        raise SystemExit('REFUSED: no read from %s' % ', '.join('region %d reader %s' % m for m in missing))
    kept, unpaired = [], []
    for k in sorted(byreg):
        A, B = byreg[k]['a'], byreg[k]['b']
        pairs = sorted(((iou(x['box'], y['box']), i, j) for i, x in enumerate(A) for j, y in enumerate(B)
                        if iou(x['box'], y['box']) >= IOU and agree(x, y)), reverse=True)
        ua, ub = set(), set()
        for v, i, j in pairs:
            if i in ua or j in ub:
                continue
            ua.add(i)
            ub.add(j)
            x, y = A[i], B[j]
            lead = x if RANK[x['conf']] >= RANK[y['conf']] else y
            other = y if lead is x else x
            box = [int(round((p + q) / 2.0)) for p, q in zip(x['box'], y['box'])]
            alts = []
            for s in [other['char']] + list(x.get('alt') or []) + list(y.get('alt') or []):
                if score.norm_letter(s) not in (None, score.norm_letter(lead['char'])) and s not in alts:
                    alts.append(s)
            kept.append(dict(region=k, char=lead['char'], alt=alts, conf=max(x['conf'], y['conf'], key=RANK.get),
                             box=box, iou_ab=round(v, 3), core=byreg[k]['region']['core'],
                             reads=[dict(reader='a', tag=x['tag'], char=x['char'], alt=x.get('alt') or [], conf=x['conf'], box=x['box'], note=x.get('note', '')),
                                    dict(reader='b', tag=y['tag'], char=y['char'], alt=y.get('alt') or [], conf=y['conf'], box=y['box'], note=y.get('note', ''))]))
        unpaired += [x['tag'] for i, x in enumerate(A) if i not in ua] + [y['tag'] for j, y in enumerate(B) if j not in ub]
    out, deduped = [], []
    for L in sorted(kept, key=lambda L: (-RANK[L['conf']], L['region'])):
        c = ((L['box'][0] + L['box'][1]) / 2.0, (L['box'][2] + L['box'][3]) / 2.0)
        L['own_core'] = inside(c, L['core'])
        clash = [M for M in out if M['region'] != L['region'] and iou(M['box'], L['box']) >= IOU and agree(M, L)]
        if not clash:
            out.append(L)
            continue
        M = clash[0]
        if L['own_core'] and not M['own_core']:
            out[out.index(M)] = L
            deduped.append('%s kept over %s' % (L['reads'][0]['tag'], M['reads'][0]['tag']))
        else:
            deduped.append('%s kept over %s' % (M['reads'][0]['tag'], L['reads'][0]['tag']))
    return out, dropped, unpaired, deduped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--marks', required=True)
    ap.add_argument('--out', required=True)
    A = ap.parse_args(argv)
    if os.path.exists(A.out):
        raise SystemExit('REFUSED: %s exists; a reference is written once' % A.out)
    runs = json.load(open(A.marks, encoding='utf-8'))
    runs = runs.get('result', runs) if isinstance(runs, dict) else runs
    out, dropped, unpaired, deduped = merge(runs)
    letters = []
    for n, L in enumerate(sorted(out, key=lambda L: (L['box'][0], L['box'][2])), 1):
        b = L['box']
        c = [(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0]
        letters.append(dict(id='W00-%02d' % n, region=L['region'], char=L['char'], alt=L['alt'], conf=L['conf'], box=b,
                            counted_in_N_ref=L['conf'] in ('clear', 'probable'), centre_map_px=c,
                            centre_in_render_crop=0 <= c[0] < H and 0 <= c[1] < W, iou_ab=L['iou_ab'], reads=L['reads'],
                            size_mm_hw=[round((b[1] - b[0]) * PX_UM / 1000, 2), round((b[3] - b[2]) * PX_UM / 1000, 2)]))
    n_ref = sum(1 for L in letters if L['counted_in_N_ref'] and L['centre_in_render_crop'])
    doc = dict(
        source_map='geom/day1-9um/gates/s3/20260220213127-w00/ink-detection/ (the native 2.403 um tif)',
        map_shape_rows_cols=[16460, 18560], map_pixel_um=PX_UM,
        render_crop_map_px={'rows': [0, H], 'cols': [0, W], 'from': 'w00-grid.json canvas_origin_px [0, 0], pool 4, H %d, W %d' % (RH, RW)},
        map_to_render='render_i = row / 4, render_j = col / 4 (edge coordinates; ref-from-letters subtracts the half pixel)',
        box_order='[row0, row1, col0, col1] in map pixels of the w00 ink tif, half-open, row 0 at the top',
        reader='Reader A, second pass (reviewer ruling 2026-10-06): two fresh blind readers per 3 x 3 region, orientation '
               'given as upright and unmirrored, kept only where both agree (identity or a listed alternative, IoU >= 0.3)',
        orientation={'mirrored': False, 'upside_down': False, 'evidence': 'given: w00big reads and 7 of 9 first-pass regions'},
        confidence_scale={'clear': 'identity certain from the shape alone; counted in N_ref',
                          'probable': 'identity read with one plausible alternative noted; counted in N_ref',
                          'possible': 'a letter-like shape whose identity is a guess; NOT counted in N_ref'},
        rules=dict(iou=IOU, counted='clear or probable from either reader when both mark it', dedupe='by box across regions'),
        summary=dict(N_ref=sum(L['counted_in_N_ref'] for L in letters), N_ref_centre_in_render_crop=n_ref,
                     N_possible_not_counted=sum(not L['counted_in_N_ref'] for L in letters), N_all_marks=len(letters),
                     unpaired_marks=len(unpaired), dropped_marks=len(dropped), deduped=len(deduped)),
        unpaired=unpaired, dropped=dropped, deduped=deduped, letters=letters)
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, 'x', encoding='utf-8') as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print('w00ref-merge2: %d letters kept by agreement (%d counted), %d unpaired, %d dropped, %d deduped -> %s' % (
        len(letters), n_ref, len(unpaired), len(dropped), len(deduped), A.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
