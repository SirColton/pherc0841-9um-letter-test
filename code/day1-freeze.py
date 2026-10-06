"""day1-freeze.py - freeze prereg-v2's two arms from the w00big selection scores, before any 252405 map.

prereg-v2 (geom/day1-9um/prereg-v2.json, sha256 checked; a mismatch stops the run), selection_on_w00big_only:
  primary    AUCsup
  tie_break  within 0.01 AUCsup of the best: highest hp r at sigma 192 um, output blur 2 px, vs ScrollPrize's 2.4 um map
  frozen     arm A = the selected recipe (R1-R4); arm B = the best single member by the same rule
Excluded by the prereg's own words: Scheirer (reference_arm_only), Hecate 9.6 (never_in_a_decision), s43-50k
(diagnostic only), mirror-TTA rows outside R3 (tta: mirror TTA only in R3), fwd rows (order rev), per-centre and
three-centre rows (per_member_map five-centre mean); s42-75k is not a prereg member (the select part's loader check).
Reads the select part's score JSONs (s48 for AUCsup, s192 for hp r; AUCsup must agree between them), re-hashes every
file an arm names, rebuilds arm A's w00big map from its five centre maps and requires it equal the scored recipe map.
252405: its stack is stat()ed for path, size and mtime only; no byte of it is read. test/ must hold nothing but
gate1.* and labels-hf (no 252405 model map) or the run stops. Refuses to overwrite an existing freeze.
CPU only. Usage: day1-freeze.py
Writes geom/day1-9um/test/recipe-frozen.json, recipe-frozen.sha256, recipe-frozen-run.txt.
"""
import datetime
import glob
import hashlib
import json
import os

import numpy as np

import runfile

D9 = 'geom/day1-9um'
SEL = os.path.join(D9, 'select')
TEST = os.path.join(D9, 'test')
OUT = os.path.join(TEST, 'recipe-frozen.json')
SHA_OUT = os.path.join(TEST, 'recipe-frozen.sha256')
REP = os.path.join(TEST, 'recipe-frozen-run.txt')
PREREG = os.path.join(D9, 'prereg-v2.json')
PREREG_SHA = '49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c'
WINDOW = 0.01


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


if sha(PREREG) != PREREG_SHA:
    raise SystemExit('prereg-v2.json sha256 differs: stop')
if os.path.exists(OUT) or os.path.exists(SHA_OUT):
    raise SystemExit('%s or its .sha256 already exists: a freeze is never rewritten' % OUT)
present = sorted(p for p in os.listdir(TEST) if not p.startswith('recipe-frozen-run'))
if present != ['gate1.json', 'gate1.txt', 'labels-hf']:
    raise SystemExit('test/ holds %s, not only gate1.* and labels-hf: a 252405 map may exist, stop' % present)
prereg = json.load(open(PREREG, encoding='utf-8'))
runfile.claim(REP)
lines = []


def say(s=''):
    print(s, flush=True)
    lines.append(s)


# --- the select part's scores -------------------------------------------------------
FILES = ['score-members', 'score-ondisk', 'score-tta', 'score-recipes', 'score-loader-check']
rows48, rows192, src = {}, {}, {}
for f in FILES:
    for s, rows in ((48, rows48), (192, rows192)):
        p = os.path.join(SEL, '%s-s%d.json' % (f, s))
        d = json.load(open(p))
        if d['sigma_um'] != float(s):
            raise SystemExit('%s says sigma %s' % (p, d['sigma_um']))
        src[p.replace('\\', '/')] = {'sha256': sha(p), 'sigma_um': d['sigma_um'], 'n_scored': d['n_scored'],
                                     'n_core': d['n_core'], 'pos': d['pos'], 'neg_sup': d['neg_sup']}
        rows.update({r['map']: r for r in d['rows']})
for p in ('select-scores.txt', 'recipes.txt', 'loaders.txt'):
    src[os.path.join(SEL, p).replace('\\', '/')] = {'sha256': sha(os.path.join(SEL, p))}


def row(name, arm):
    a, b = rows48[name], rows192[name]
    if a['auc_sup'] != b['auc_sup'] or a['auc_all'] != b['auc_all']:
        raise SystemExit('AUC differs between sigma files for %s' % name)
    return {'row': name, 'arm': arm, 'auc_sup': a['auc_sup'], 'auc_all': a['auc_all'],
            'hp_r_192um_b2_vs_SP': b['hp_sp_b2']['r'], 'hp_r_192um_b2_null_max': b['hp_sp_b2']['null_max'],
            'hp_r_192um_b2_z': b['hp_sp_b2']['z'], 'hp_r_192um_b2_n_null_ge': b['hp_sp_b2']['n_null_ge'],
            'hp_r_48um_b2_vs_SP': a['hp_sp_b2']['r'], 'hp_r_48um_b2_null_max': a['hp_sp_b2']['null_max']}


def apply_rule(cands):
    best = max(c['auc_sup'] for c in cands)
    for c in cands:
        c['auc_sup_below_best'] = best - c['auc_sup']
        c['in_window'] = c['auc_sup'] >= best - WINDOW
    win = [c for c in cands if c['in_window']]
    top = max(c['hp_r_192um_b2_vs_SP'] for c in win)
    pick = [c for c in win if c['hp_r_192um_b2_vs_SP'] == top]
    if len(pick) != 1:
        raise SystemExit('exact tie on the tie-break among %s: the prereg is silent, stop' % [c['row'] for c in pick])
    for c in cands:
        c['selected'] = c is pick[0]
    return pick[0], best


say('day1-freeze.py  prereg-v2 sha256 %s (matches)' % PREREG_SHA)
say('rule: primary AUCsup; within %.2f of the best, highest hp r 192 um blur 2 vs ScrollPrize 2.4 um' % WINDOW)
say('')

RECIPES = {'R1': 'best single claim-eligible-or-on-disk member (five-centre mean)',
           'R2': 'rank-normalised mean of the three MIT community members',
           'R3': 'R2 with mirror TTA',
           'R4': 'R2 plus soup42 and soup43, rank-normalised mean'}
rc = [row('%s rev c+0' % k, 'RECIPE') for k in RECIPES]
for c, k in zip(rc, RECIPES):
    c['recipe'] = k
    c['prereg_definition'] = prereg['recipes'][k]
selA, bestA = apply_rule(rc)
say('ARM A, recipes R1-R4 (w00big, rev, five-centre mean): AUCsup | hp r 192 b2 [null max]')
for c in sorted(rc, key=lambda c: -c['auc_sup']):
    say('  %s  %.4f (best - %.4f) | %+.4f [%+.4f]%s%s' % (
        c['recipe'], c['auc_sup'], c['auc_sup_below_best'], c['hp_r_192um_b2_vs_SP'], c['hp_r_192um_b2_null_max'],
        '  in window' if c['in_window'] else '  OUT of window', '  <- SELECTED' if c['selected'] else ''))
say('')

MEMBERS = [('dense_native', 'MIT claim-eligible'), ('klavis_dense', 'MIT claim-eligible'),
           ('tauil_ft6k', 'MIT claim-eligible'), ('s43_75k', 'on-disk'), ('soup42', 'on-disk'), ('soup43', 'on-disk')]
mc = [row('%s rev mean5' % n, a) for n, a in MEMBERS]
for c, (n, _) in zip(mc, MEMBERS):
    c['member'] = n
selB, bestB = apply_rule(mc)
say('ARM B, single members (w00big, rev, five-centre mean): AUCsup | hp r 192 b2 [null max]')
for c in sorted(mc, key=lambda c: -c['auc_sup']):
    say('  %-13s %.4f (best - %.4f) | %+.4f [%+.4f]%s%s' % (
        c['member'], c['auc_sup'], c['auc_sup_below_best'], c['hp_r_192um_b2_vs_SP'], c['hp_r_192um_b2_null_max'],
        '  in window' if c['in_window'] else '', '  <- SELECTED' if c['selected'] else ''))
say('')

# --- recorded, never eligible ---------------------------------------------------------
EXCL = [('scheirer_ft rev mean5', 'Scheirer: prereg members.reference_arm_only'),
        ('s43_50k rev mean5', 'prereg members.on_disk: "diagnostic only"'),
        ('tauil_ft6k_tta rev mean5', 'prereg inference.tta: "mirror TTA only in R3"'),
        ('dense_native_tta rev mean5', 'prereg inference.tta: "mirror TTA only in R3"'),
        ('klavis_dense_tta rev mean5', 'prereg inference.tta: "mirror TTA only in R3"'),
        ('s42_75k rev mean5', 'not a prereg member: the select part ran it as a loader check')]
ex = [row(n, why) for n, why in EXCL]
# what the rule would have said had an excluded row been let in: recorded so the exclusion's weight is visible
cf = {}
for label, extra in (('arm B if mirror-TTA single members were eligible', ex[2:5]),
                     ('arm B if Scheirer were eligible', ex[:1]),
                     ('arm A if Scheirer were a recipe', ex[:1])):
    base = rc if label.startswith('arm A') else mc
    pool = [dict(c) for c in base] + [dict(c) for c in extra]
    w, b = apply_rule(pool)
    cf[label] = {'would_select': w['row'], 'auc_sup': w['auc_sup'], 'hp_r_192um_b2_vs_SP': w['hp_r_192um_b2_vs_SP'],
                 'best_auc_sup': b, 'window': [c['row'] for c in pool if c['in_window']]}
say('never eligible, recorded only:')
for c in ex:
    say('  %-27s %.4f | %+.4f   (%s)' % (c['row'], c['auc_sup'], c['hp_r_192um_b2_vs_SP'], c['arm']))
for k, v in cf.items():
    say('  counterfactual, NOT used: %s -> %s (AUCsup %.4f, hp r %+.4f)' % (k, v['would_select'], v['auc_sup'], v['hp_r_192um_b2_vs_SP']))
say('')

# --- the arms' files ------------------------------------------------------------------
if selA['recipe'] != 'R1':
    raise SystemExit('arm A is %s: this script only knows how to pin R1\'s single member; extend it before freezing' % selA['recipe'])
r1_line = [l for l in open(os.path.join(SEL, 'recipes.txt'), encoding='utf-8') if l.startswith('R1 = ')]
r1_member = r1_line[0].split()[2]
if r1_member != selB['member']:
    raise SystemExit('recipes.txt built R1 from %s but the rule here picks %s: stop' % (r1_member, selB['member']))
CK = {'tauil_ft6k': ('geom/ckpt-community/ink9um_native0139_ft6k.pth',
                     'TAUIL native0139 ft6k (MIT)',
                     'TAUIL-Abd-Elilah/cross-scan-ink-transfer v1.0 (GitHub, MIT)',
                     "payload['ema_model'] moved to 'model' (TAUIL's ft9.py writes the plain fine-tuned state_dict there); "
                     'strict load into koine make_model: 0 missing, 0 unexpected; 508 tensors; 68,175,426 parameters; '
                     '||w - released s42-75k|| / ||s42-75k|| = 0.0211')}
member = selB['member']
if member not in CK:
    raise SystemExit('no checkpoint record for %s' % member)
cpath, pkey, origin, transform = CK[member]
csha = sha(cpath)
if csha != prereg['members']['community_claim_eligible'][pkey]:
    raise SystemExit('%s sha256 %s differs from prereg-v2' % (cpath, csha))
ppath = os.path.join(SEL, 'ckpt', member + '.pth').replace('\\', '/')
psha = sha(ppath)
logged = [l for l in open(os.path.join(SEL, 'loaders.txt'), encoding='utf-8') if ('ckpt\\%s.pth' % member) in l or ('ckpt/%s.pth' % member) in l]
if not logged or psha not in logged[0]:
    raise SystemExit('%s sha256 %s is not the one loaders.txt wrote' % (ppath, psha))
CENTRES = prereg['inference']['depth_centres_layers']
maps = []
for dc in CENTRES:
    p = os.path.join(SEL, 'maps', '%s-c%+d-rev.npy' % (member, dc)).replace('\\', '/')
    maps.append({'centre': dc, 'path': p, 'sha256': sha(p)})
m5 = np.mean([np.load(m['path']).astype(np.float32) for m in maps], 0)
rpath = os.path.join(SEL, 'recipes', 'R1-c+0-rev.npy').replace('\\', '/')
r1 = np.load(rpath)
diff = float(np.abs(m5 - r1).max())
if m5.shape != r1.shape or diff != 0.0:
    raise SystemExit('R1 map is not the mean5 of %s (max |diff| %g): stop' % (member, diff))
say('arm A = arm B = %s; R1 map rebuilt from its five rev centre maps: shape %s, max |diff| %g' % (member, r1.shape, diff))
say('  source %s sha256 %s (= prereg-v2)' % (cpath, csha))
say('  payload %s sha256 %s (= loaders.txt)' % (ppath, psha))

code = {}
for p in ('day1-ink9-maps.py', 'day1-select-loaders.py', 'day1-select-recipes.py', 'day1-score-baselines.py',
          'day1-select-table.py', 'polite-run.py', 'day1-freeze.py',
          'koine/ink-detection/koine_machines/inference/infer.py'):
    code[p] = sha(p)
pk = sorted(glob.glob('koine/ink-detection/koine_machines/**/*.py', recursive=True))
h = hashlib.sha256()
for p in pk:
    h.update(p.replace('\\', '/').encode() + b'\0' + sha(p).encode() + b'\n')
code['koine/ink-detection/koine_machines/**/*.py (sorted path\\0sha256 lines, %d files)' % len(pk)] = h.hexdigest()

st = os.stat(os.path.join(D9, 'render', '252405-stack.npy'))
stack252405 = {'path': D9 + '/render/252405-stack.npy', 'bytes': st.st_size,
               'mtime': datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec='seconds'),
               'shape_per_render_log': [41, 2125, 2265],
               'not_hashed': 'conservative reading of "do not run anything on 252405": stat only, no byte read'}

INFER = {
    'driver': 'day1-ink9-maps.py (sha256 above), the recipe every w00big member row was scored with',
    'slab': 'stack[c - 10 : c + 11] (21 layers) about c = n_layers // 2 + centre',
    'centres_layers': CENTRES,
    'order': prereg['inference']['order'],
    'order_applied': 'rev: the 21-layer slab reversed (s21[::-1]) before writing the zarr',
    'zarr': 'v2, uint8, chunks (21, 128, 128)',
    'windows': [[0, 17], [2, 19], [4, 21]],
    'infer_argv_per_window': '<zarr> <payload> <out.tif> --layer-start a --layer-end b --no-compile --num-workers 0',
    'infer_defaults_used': 'overlap 0.5 (DEFAULT_OVERLAP), blend auto -> Hann, batch-size 1, no TTA',
    'from_checkpoint_config': 'model vesuvius_unet_3d_stem_2d, in_chans 17, roi 128, preprocessing tifxyz_robust '
                              '(robust-MAD per patch, 1/99), AMP fp16',
    'per_centre_map': 'mean of the three uint8 window outputs, float32 (LANCZOS resize only if infer returns another shape)',
    'per_member_map': 'arithmetic five-centre mean, float32 (' + prereg['inference']['per_member_map'] + ')',
    'tta': 'none',
    'output_normalisation': 'none: the raw five-centre mean, exactly as R1 and the member row were scored; '
                            'no rank normalisation (that belongs to R2-R4)',
    'on_252405': 'the same recipe with only the input stack changed to %s; centre index 41 // 2 = 20, so absolute '
                 'centres 16, 18, 20, 22, 24 and slabs [6,27), [8,29), [10,31), [12,33), [14,35). '
                 'day1-ink9-maps.py hardcodes the w00big STACK: the 252405 run needs an added CLI flag for the stack '
                 'and nothing else changed' % stack252405['path'],
}


def arm(label, rule_row, kind):
    return {
        'label': label,
        'chosen_by': kind,
        'recipe': 'R1' if kind == 'selected recipe' else None,
        'members': [{'name': member, 'prereg_key': pkey, 'origin': origin,
                     'source_checkpoint': {'path': cpath, 'sha256': csha, 'prereg_sha256_match': True},
                     'koine_payload': {'path': ppath, 'sha256': psha, 'built_by': 'day1-select-loaders.py',
                                       'transform': transform}}],
        'combination': 'single member, no ensemble',
        'centres_layers': CENTRES,
        'order': 'rev',
        'tta': 'none',
        'normalisation': INFER['output_normalisation'],
        'w00big_selection_maps': {'centre_maps': maps,
                                  'five_centre_mean': {'path': rpath, 'sha256': sha(rpath),
                                                       'equals_mean_of_centre_maps': True, 'max_abs_diff': diff}},
        'w00big_score_row': rule_row,
    }


now = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
doc = {
    'name': 'day1 frozen recipe, prereg-v2 selection on w00big',
    'written': now,
    'prereg': {'path': PREREG, 'sha256': PREREG_SHA,
               'notes_seen': {p: sha(os.path.join(D9, p)) for p in ('prereg-v2-note-1.txt', 'prereg-v2-note-2.txt',
                                                                    'prereg-v2-note-3.txt')},
               'notes_change_selection': False},
    'rule': prereg['selection_on_w00big_only'],
    'rule_applied': {'primary': 'auc_sup (from the s48 files; equal in the s192 files, checked)',
                     'window': 'auc_sup >= best - 0.01, inclusive',
                     'tie_break': 'hp_sp_b2 r from the sigma 192 um files (ScrollPrize 2.4 um map key, blur 2 px)',
                     'exact_tie_on_tie_break': 'none occurred (the prereg is silent; the script would stop)'},
    'arms': {
        'A_selected_recipe': arm('selected recipe', selA, 'selected recipe'),
        'B_best_single_member': dict(arm('best single member', selB, 'best single member'),
                                     identical_to_A=True,
                                     identity_note='R1 won the recipe rule, and R1 is by definition the best single '
                                                   'member, so both arms are the same member, map and recipe. One '
                                                   '252405 computation serves both; report it once under both labels. '
                                                   'The guard the prereg meant arm B to be (ensemble letter-softening) '
                                                   'has nothing to guard here: arm A is not an ensemble.'),
    },
    'decided_by': {
        'arm_A_rows': rc, 'arm_A_best_auc_sup': bestA,
        'arm_B_rows': mc, 'arm_B_best_auc_sup': bestB,
        'never_eligible_rows': ex,
        'counterfactuals_not_used': cf,
    },
    'inference': INFER,
    'input_252405': stack252405,
    'state_at_freeze': {
        'test_dir_listing': present,
        'no_252405_model_map': 'test/ holds only gate1.* and labels-hf; render/GATE2-SKIPPED.txt says no model has run '
                               'on the 252405 render; day1-render-gate2.py has never run past its baselines/DONE check',
        'this_part_read_of_252405': 'one os.stat of the stack; nothing else',
    },
    'score_sources': src,
    'code_sha256': code,
    'conservative_readings': [
        'arm B pool = the prereg\'s claim-eligible members plus its on-disk members except s43-50k, which the prereg '
        'calls diagnostic only',
        'mirror-TTA single rows are not arm B candidates: the prereg puts mirror TTA in R3 only',
        'arm B is frozen as the rule gives it, the same member as arm A; no runner-up is substituted',
        'the 252405 stack is not hashed (stat only), reading "do not run anything on 252405" literally',
        'the inference recipe is day1-ink9-maps.py\'s, unchanged, as every scored w00big row used it',
    ],
}
text = json.dumps(doc, indent=1, ensure_ascii=False) + '\n'
with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)
fsha = sha(OUT)
with open(SHA_OUT, 'w', encoding='utf-8', newline='\n') as f:
    f.write('%s  recipe-frozen.json\n' % fsha)
say('')
say('wrote %s at %s' % (OUT, now))
say('sha256 %s  -> %s' % (fsha, SHA_OUT))
runfile.done(REP, '\n'.join(lines) + '\n')
