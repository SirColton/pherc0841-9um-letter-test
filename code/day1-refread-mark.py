# Reader A: turn the hand marks on ScrollPrize's 2.4 um map of 252405 into letters.json and marked PNGs.
# Reads only render/src/inkmap.tif (the 2.4 um map) and reference/marks-hand.json.
import json, os, numpy as np, tifffile
from PIL import Image, ImageDraw, ImageFont
BASE = 'geom/day1-9um'
MAP = BASE + '/render/src/inkmap.tif'
REF = BASE + '/reference'
PX_UM = 2.403            # map pixel pitch (2.403 um volume 20260319124803)
R0, C0, POOL, RH, RW = 840, 4200, 4, 2125, 2265   # 252405-grid.json: render (i,j) = canvas block [R0+4i,+4) x [C0+4j,+4)
THR = 128
a = tifffile.imread(MAP)
H, W = a.shape
spec = json.load(open(REF + '/marks-hand.json', encoding='utf-8'))
crop = [R0, R0 + POOL * RH, C0, C0 + POOL * RW]
out = []
for L in spec['letters']:
    r0, r1, c0, c1 = L['box']
    sub = a[r0:r1, c0:c1].astype(np.float32)
    ink = sub > THR
    rr, cc = np.nonzero(ink)
    cen = [(r0 + r1) / 2, (c0 + c1) / 2]
    if len(rr):
        wgt = sub[rr, cc]
        cent = [float(r0 + (rr * wgt).sum() / wgt.sum()), float(c0 + (cc * wgt).sum() / wgt.sum())]
        ibb = [int(r0 + rr.min()), int(r0 + rr.max() + 1), int(c0 + cc.min()), int(c0 + cc.max() + 1)]
    else:
        cent, ibb = None, None
    inside_c = crop[0] <= cen[0] < crop[1] and crop[2] <= cen[1] < crop[3]
    inside_b = crop[0] <= r0 and r1 <= crop[1] and crop[2] <= c0 and c1 <= crop[3]
    d = dict(L)
    d.update({
        'counted_in_N_ref': L['conf'] in ('clear', 'probable'),
        'centre_map_px': [round(cen[0], 1), round(cen[1], 1)],
        'ink_centroid_map_px': [round(cent[0], 1), round(cent[1], 1)] if cent else None,
        'ink_bbox_map_px_thr128': ibb,
        'ink_frac_in_box_thr128': round(float(ink.mean()), 3),
        'size_mm_hw': [round((r1 - r0) * PX_UM / 1000, 2), round((c1 - c0) * PX_UM / 1000, 2)],
        'centre_render_px_9p61um': [round((cen[0] - R0) / POOL, 1), round((cen[1] - C0) / POOL, 1)],
        'box_render_px_9p61um': [round((r0 - R0) / POOL, 1), round((r1 - R0) / POOL, 1), round((c0 - C0) / POOL, 1), round((c1 - C0) / POOL, 1)],
        'centre_in_render_crop': bool(inside_c),
        'box_fully_in_render_crop': bool(inside_b),
    })
    out.append(d)
cnt = [d for d in out if d['counted_in_N_ref']]
summary = {
    'N_ref': len(cnt),
    'N_ref_centre_in_render_crop': sum(d['centre_in_render_crop'] for d in cnt),
    'N_possible_not_counted': len(out) - len(cnt),
    'N_all_marks': len(out),
    'N_all_marks_centre_in_render_crop': sum(d['centre_in_render_crop'] for d in out),
    'N_ref_below_15': len(cnt) < 15,
}
doc = {
    'source_map': MAP,
    'source_map_note': 'ScrollPrize ink detection, PHerc0841 auto_grown_20260220174252405 (S3 20260221022814), 2.403 um volume 20260319124803, new_canon_autoresearch_recipe tile256 stride128; byte-identical to gates/s3/.../ink-detection/*.tif',
    'map_shape_rows_cols': [H, W],
    'map_pixel_um': PX_UM,
    'render_crop_map_px': {'rows': crop[:2], 'cols': crop[2:], 'from': 'render/252405-grid.json canvas_origin_px [840,4200], pool 4, H 2125, W 2265'},
    'map_to_render': 'render_i = (row - 840) / 4, render_j = (col - 4200) / 4 (render px 9.612 um)',
    'match_rule_reminder': 'prereg: same identity AND centre within 0.5 mm = 208 map px = 52 render px',
    'orientation': spec.get('orientation'),
    'reader': spec['reader'], 'box_order': spec['box_order'], 'confidence_scale': spec['confidence_scale'],
    'summary': summary,
    'letters': out,
}
json.dump(doc, open(REF + '/letters.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summary))
for d in out:
    print(d['id'], d['conf'], 'centre', d['centre_map_px'], 'centroid', d['ink_centroid_map_px'], 'render', d['centre_render_px_9p61um'], 'in_crop', d['centre_in_render_crop'], 'ink', d['ink_frac_in_box_thr128'])


def font(sz):
    for f in ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf'):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def draw(ds, r_lo, r_hi, c_lo, c_hi, name, fs):
    sub = a[r_lo:r_hi, c_lo:c_hi]
    h, w = (sub.shape[0] // ds) * ds, (sub.shape[1] // ds) * ds
    p = sub[:h, :w].reshape(h // ds, ds, w // ds, ds).astype(np.float32).mean((1, 3))
    im = Image.fromarray(np.clip(p * 0.8, 0, 255).astype(np.uint8)).convert('RGB')
    dr = ImageDraw.Draw(im)
    F = font(fs)
    Fs = font(max(12, fs // 2))

    def xy(r, c):
        return ((c - c_lo) / ds, (r - r_lo) / ds)
    x0, y0 = xy(crop[0], crop[2])
    x1, y1 = xy(crop[1], crop[3])
    dr.rectangle([x0, y0, x1, y1], outline=(200, 200, 200), width=1)
    dr.text((x0 + 4, y0 + 4), '9 um render crop', fill=(200, 200, 200), font=Fs)
    for d in out:
        r0, r1, c0, c1 = d['box']
        bx0, by0 = xy(r0, c0)
        bx1, by1 = xy(r1, c1)
        col = (40, 140, 255) if d['counted_in_N_ref'] else (255, 150, 0)
        wdt = 4 if d['conf'] == 'clear' else 2
        if d['counted_in_N_ref']:
            dr.rectangle([bx0, by0, bx1, by1], outline=col, width=wdt)
        else:
            step = 8
            for x in np.arange(bx0, bx1, step * 2):
                dr.line([(x, by0), (min(x + step, bx1), by0)], fill=col, width=2)
                dr.line([(x, by1), (min(x + step, bx1), by1)], fill=col, width=2)
            for y in np.arange(by0, by1, step * 2):
                dr.line([(bx0, y), (bx0, min(y + step, by1))], fill=col, width=2)
                dr.line([(bx1, y), (bx1, min(y + step, by1))], fill=col, width=2)
        cx, cy = xy(*d['centre_map_px'])
        dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=col)
        lab = d['char'] + ('' if d['counted_in_N_ref'] else '?')
        dr.text((bx0 + 2, by0 - fs - 2), lab, fill=col, font=F)
        dr.text((bx0 + 2, by1 + 1), d['id'], fill=col, font=Fs)
    leg = ('Reader A, 2.4 um map only. Solid blue = counted (thick: clear, thin: probable). '
           'Dashed orange = possible, not counted. N_ref = %d (%d inside the 9 um crop)'
           % (summary['N_ref'], summary['N_ref_centre_in_render_crop']))
    dr.rectangle([0, im.height - fs - 8, im.width, im.height], fill=(0, 0, 0))
    dr.text((4, im.height - fs - 6), leg, fill=(230, 230, 230), font=Fs)
    im.save(REF + '/' + name)
    print(REF + '/' + name, im.size)


draw(10, 0, H, 0, W, 'letters-marked.png', 26)
draw(5, 1800, 10000, 3800, 17800, 'letters-marked-text.png', 40)
