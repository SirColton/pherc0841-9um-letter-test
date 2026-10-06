"""day1-select-table.py - one table of every member and recipe scored on w00big, for part "select".

Reads day1-score-baselines.py JSONs written at --sigma 48 and --sigma 192 (pairs given on the command
line; rows are matched by map name, so every pair must come from the same maps dir) and writes
geom/day1-9um/select/select-scores.txt, then appends geom/day1-9um/select/select-notes.txt verbatim (the
loader proofs, what was not done, and provenance notes, written by hand from loaders.txt and recipes.txt).
Prints numbers only: the pick is the next part's.
CPU only. Usage: day1-select-table.py <s48.json>:<s192.json> [...]
"""
import hashlib
import json
import os
import sys

import runfile

D9 = 'geom/day1-9um'
PREREG_SHA = '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'
if hashlib.sha256(open(os.path.join(D9, 'prereg-v2.json'), 'rb').read()).hexdigest() != PREREG_SHA:
    raise SystemExit('prereg-v2.json sha256 differs: stop')
OUTF = os.path.join(D9, 'select', 'select-scores.txt')
runfile.claim(OUTF)
rows = {}
meta = []
for pair in sys.argv[1:]:
    f48, f192 = pair.split(':', 1) if pair.count(':') == 1 else pair.rsplit(':', 1)
    j48, j192 = json.load(open(f48)), json.load(open(f192))
    assert abs(j48['sigma_um'] - 48) < 1e-9 and abs(j192['sigma_um'] - 192) < 1e-9, pair
    meta.append((f48, f192, j48, j192))
    r192 = {r['map']: r for r in j192['rows']}
    for r in j48['rows']:
        rows.setdefault(r['map'], (r, r192[r['map']], os.path.dirname(f48)))

ARM = {'dense_native': 'MIT', 'klavis_dense': 'MIT', 'tauil_ft6k': 'MIT', 'scheirer_ft': 'ref-arm',
       'dense_native_tta': 'MIT+TTA', 'klavis_dense_tta': 'MIT+TTA', 'tauil_ft6k_tta': 'MIT+TTA',
       's43_75k': 'on-disk', 'soup42': 'on-disk', 'soup43': 'on-disk', 's43_50k': 'diag only', 's42_75k': 'loader-chk'}
NOTES = os.path.join(D9, 'select', 'select-notes.txt')   # hand-written provenance block, printed verbatim


def arm(name):
    if name.startswith('REF'):
        return 'reference'
    if name.startswith(('ENS', 'RANK')):
        return 'extra'
    if name.split()[0] in ('R1', 'R2', 'R3', 'R4'):
        return 'RECIPE'
    if name.startswith('CHECKens2'):
        return 'check'
    return ARM.get(name.split()[0], '?')


def cell(h):
    return '%+.3f%s[%+.3f]' % (h['r'], ' ' if h['r'] > h['null_max'] else '*', h['null_max'])


L = []
L.append('select-scores.txt  part "select", day-1 plan, prereg-v2 sha256 %s' % PREREG_SHA)
L.append('w00big only (native 9.366 um render of PHerc0841 w00, 995x1245 at 9.612 um); nothing of 252405 was opened.')
j = meta[0][2]
L.append('pixels: scored %d, pos %d, neg(sup) %d, neg(all) %d; hp core %d at 48 um, %d at 192 um' % (
    j['n_scored'], j['pos'], j['neg_sup'], j['neg_all'], j['n_core'], meta[0][3]['n_core']))
L.append('labels: HF inklabels_v2 + supervision_mask_v2 on the w00 grid. AUCsup = pos label>=0.75 vs neg label<=0 AND sup>=0.75.')
L.append('hp r: Scheirer letter-scale score vs the ScrollPrize 2.4 um w00 map (ds8 key), read blurred b0/b2/b4 px;')
L.append('  [x] = max of the 48 rolled-key nulls (1-3 mm shifts); * = r NOT above that null max. h24 = key is our Hecate 2.4 um map.')
L.append('every member row: rev order, mean of five depth centres (-4,-2,0,+2,+4 layers), each centre the mean of')
L.append('  windows 0-17, 2-19, 4-21 of a 21-layer stack (day1-ink9-maps.py). fwd rows: forward-order controls.')
L.append('arms: MIT = claim-eligible community; ref-arm = Scheirer, reference only; on-disk = released/soups;')
L.append('  diag only = s43-50k; RECIPE = prereg R1-R4 (recipes.txt says how each was built); extra = the scorer\'s own')
L.append('  ENS/RANK rows, not prereg recipes; check = rebuild of the scorer\'s ENS soup42+soup43 (must equal it).')
L.append('sources: %s' % '; '.join('%s + %s' % (a, b) for a, b, _, _ in meta))
L.append('')
hdr = '%-44s %-9s %6s %6s | %-17s %-17s %-17s | %-17s %-17s %5s %-17s | %-17s' % (
    'map', 'arm', 'AUCsup', 'AUCall', '48um b0', '48um b2', '48um b4', '192um b0', '192um b2', 'z b2', '192um b4', 'h24 192 b2')


def line(name):
    a, b, _ = rows[name]
    shown = name.replace(' rev c+0', ' (recipe map)') if arm(name) in ('RECIPE', 'check') else name
    return '%-44s %-9s %6.4f %6.4f | %s %s %s | %s %s %5.1f %s | %s' % (
        shown[:44], arm(name), a['auc_sup'], a['auc_all'], cell(a['hp_sp_b0']), cell(a['hp_sp_b2']), cell(a['hp_sp_b4']),
        cell(b['hp_sp_b0']), cell(b['hp_sp_b2']), b['hp_sp_b2']['z'], cell(b['hp_sp_b4']), cell(b['hp_h24_b2']))


def section(title, names):
    names = [n for n in names if n in rows]
    if not names:
        return
    L.append(title)
    L.append(hdr)
    for n in names:
        L.append(line(n))
    L.append('')


mean5 = sorted([n for n in rows if n.endswith(' rev mean5') and not n.startswith(('ENS', 'RANK')) and arm(n) != 'loader-chk'],
               key=lambda n: -rows[n][0]['auc_sup'])
section('TABLE 1. members (rev, five-centre mean), recipes, controls, references - sorted within block by AUCsup',
        mean5 + sorted([n for n in rows if arm(n) == 'RECIPE'], key=lambda n: n.split()[0]) +
        sorted([n for n in rows if n.endswith(' fwd mean5')], key=lambda n: -rows[n][0]['auc_sup']) +
        [n for n in rows if arm(n) == 'reference'])
section('TABLE 2. per-centre rows and three-centre means (same scoring)',
        sorted([n for n in rows if (' c' in n or 'mean3' in n) and arm(n) not in ('RECIPE', 'check', 'loader-chk')],
               key=lambda n: (n.split()[0], n.split()[1], n.split()[2])))
section('TABLE 3. the scorer\'s own extras (not prereg recipes) and the rank-copy check',
        [n for n in rows if arm(n) in ('extra', 'check')])
section('TABLE 4. loader check only, not a prereg member: released ink_9um seed42 step-075000, the stated parent of tauil_ft6k and scheirer_ft',
        sorted([n for n in rows if arm(n) == 'loader-chk'], key=lambda n: (n.split()[1], n.split()[2])))
if os.path.exists(NOTES):
    L.append(open(NOTES, encoding='utf-8').read().rstrip())
runfile.done(OUTF, '\n'.join(L) + '\n')
print('\n'.join(L))
