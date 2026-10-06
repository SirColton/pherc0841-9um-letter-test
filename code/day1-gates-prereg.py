"""day1-gates-prereg.py - plan step 0 (d): write geom/day1-9um/gates/prereg.json and print its sha256.

The pre-registration of the USE WHAT EXISTS day-1 test (design.json, key "choice"): members, depth centres,
stack order, TTA, the four named recipes, the selection rule, metrics, thresholds, the letter match rule and the
bar rule. It is written BEFORE any community checkpoint is fetched, before any model runs on 252405, and before
252405 is rendered at 9 um. Gate results (a)-(c) are cited from their run files, whose sha256 is computed here.
Fields marked "machine_fill_in" are details the plan left open, fixed here by the machine; they are not mine
and not the judges'. Refuses to overwrite an existing prereg.json (a prereg is written once).
CPU only; writes prereg.json and prereg.sha256 under geom/day1-9um/gates/ only.
"""
import datetime, hashlib, json, os, sys

G = 'geom/day1-9um/gates'
OUT = os.path.join(G, 'prereg.json')
PLAN = 'design.json'  # the day-1 plan file (not published)
if os.path.exists(OUT):
    sys.exit('prereg.json exists; a prereg is written once. Delete it by hand only if it was never used.')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

CK = 'geom/ink9um/seed%d-step-%06d.pth'
SOUP_STEPS = (40000, 50000, 60000, 75000)
INK252 = 's3/20260221022814-auto_grown_20260220174252405/ink-detection/PHerc0841-20260221022814-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif'
INKW00 = 's3/20260220213127-w00/ink-detection/PHerc0841-20260220213127-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif'

now = datetime.datetime.now().astimezone()
P = {
 'title': 'Day 1, USE WHAT EXISTS: do existing 9 um ink readers read unseen PHerc0841 text (auto_grown_20260220174252405) from the native 9.366 um scan?',
 'written_local': now.isoformat(timespec='seconds'),
 'written_by': 'machine (Claude, AI assistant); not me, not the judges',
 'plan': {'file': PLAN, 'key': 'choice', 'sha256': sha(PLAN)},
 'frozen_before': ['any community checkpoint download', 'any 9 um model run on 252405', 'any 9 um render of 252405',
                   'any recipe selection score on w00big'],

 'gates_step0': {
  'a_eligibility': {'source': 'https://raw.githubusercontent.com/ScrollPrize/villa/main/scrollprize.org/src/data/prizeEligibility.json',
                    'fetched_utc': '2026-10-05T18:52:26Z', 'file': G + '/prizeEligibility.json', 'sha256': sha(G + '/prizeEligibility.json'),
                    'last_commit_touching_file': '75c79ac5f506 2026-09-24T21:46:30Z "scrollprize.org: prize updates (#1887)"',
                    'PHerc1203_20250820131727': 'first-letters-2027 YES, grand-prize-2027 YES',
                    'PHerc0846A_20250728152254': 'first-letters-2027 YES, grand-prize-2027 not listed',
                    'PHerc1447': 'first-letters-2027 not listed (delisted), grand-prize-2027 listed',
                    'result': 'PASS'},
  'b_w00_provenance': {
   'catalog': 'live open-data metadata.json, Last-Modified 2026-10-05 15:58:43 GMT, sha256 ' + sha(G + '/catalog-live.json'),
   'w00_catalog_id': 'segment 20260220213127-w00, original_volume_id 20250821151531, derived_from volume 20250821151531, process surface_tracing, source s3://philodemos/giorgio/PHerc0841/9um/sean_latest/w00/',
   'volume': '20250821151531 = 20250821151531-9.366um-1.2m-113keV-masked.zarr, scan 20250719115350, radix SCROLLS_HEL_4.683um_113keV_1.2m_binmean_2_PHerc_0841_HA, 113 keV, 1.2 m, 9.366 um',
   'label_meta_string': "'4.681um_113keV_1.2m_binmean_2_PHerc_0841' matches NO catalog scan (0841's radix says 4.683); 4.681 is the eligible scans' prefix (1203, 0846A: 9.362 um). The only 113 keV 0841 volume in the catalog is 20250821151531. Reading it as that volume is the machine's inference.",
   'grower_meta': 'tifxyz_original area_cm2/area_vx2 gives 9.362 um voxels for w00, 552896 and 252405 (vc_grow_seg_from_seed, PHerc0841.volpkg normal grids); a 2.403 um growth would give 2.403',
   'geometry': 'normalized 9.366-frame mesh carried by the catalog matrix lies on the published 2.403 tifxyz: point-to-plane median 0.88 (w00), 0.81 (252405), 0.89 (552896) 2.403-um voxels',
   'meaning': 'All three PHerc0841 label surfaces were traced on the 113 keV 1.2 m binmean-2 scan (9.366 um) and transformed into the 2.403 frame. The 2.403 tifxyz is a derived copy. The line in c0841-9um-run.txt "w00 was segmented on the 2.403 um scan" is wrong (reported, not edited). The test therefore runs on surfaces derived from the same scan TYPE as the eligible volumes (113 keV, 1.2 m, 2x binned BM18 helical, July 2025), but PHerc0841 is not itself eligible, and its pixel is 9.366 um against the eligible 9.362 um.',
   'result': 'SETTLED: grown on 20250821151531 (9.366 um, 113 keV)'},
  'c_252405_independence': {
   'map_fetched': {'file': G + '/' + INK252, 'bytes': os.path.getsize(os.path.join(G, INK252)), 'sha256': sha(os.path.join(G, INK252)),
                   'source': 'ScrollPrize open-data S3, PHerc0841/segments/20260221022814-auto_grown_20260220174252405/ink-detection/ (model 20260417190342 new_canon_autoresearch_recipe, tile256 stride128), on the published 2.403 tifxyz grid (733x955 x20 = 14660x19100)',
                   'note': "The HF scrollprize bucket's ink/841/auto_grown_20260220174252405/preds holds three other maps (53-85 MB); none is ~23 MB. The ~23 MB map is this one."},
   'mesh_fetched': '20260221022814-on-20260319124803-2.403um.tifxyz, -on-20250821151531-9.366um.tifxyz, tifxyz_original, tifxyz_normalized (S3)',
   'labels_fetched': 'S3 ink-labels/2.403um-volume-20260319124803/20260918 (inklabels, supervision, validation) for w00, 552896, 252405',
   'co_supervised_within_80_vox': {'252405_vs_w00': 0, '252405_vs_552896': 0},
   'min_supervised_to_supervised_vox': {'w00': {'S3_20260918': 82.6, 'HF_v2': 84.6}, '552896': {'S3_20260918': 142.8, 'HF_v2': 141.9}},
   'min_252405_inklabel_to_any_w00_vertex_vox': 82.6,
   'min_inklabel_to_inklabel_vox': {'w00': 297.4, '552896': 150.0},
   'w00big_selection_frame_to_252405': 'all vertices min 56.2 vox (479 within 80); supervised and ink-labelled min 82.6 vox, 0 within 80',
   'geometry_caveat': '5452 of 522534 252405 vertices (about 0.13 cm2) lie within 15 vox (36 um) of w00, all outside 252405 supervised cells; the 80-vox margin to w00 is 2.6 vox (6 um).',
   'result': 'PASS (zero co-supervised points within 80 voxels, both label sets), narrow margin to w00'},
  'run_files': {f: sha(os.path.join(G, f)) for f in ('provenance-run.txt', 'overlap-run.txt', 'calib-run.txt')},
  'scripts': {f: sha(f) for f in ('day1-gates-provenance.py', 'day1-gates-overlap.py', 'day1-gates-calib.py')},
 },

 'reference_calibration_at_prereg_time': {
  'what': 'ScrollPrize 2.4 um map (the hp r key) scored against human labels, AUC with supervised negatives. Measured on the 2.4 um reference only; no 9 um output was looked at.',
  '252405': {'HF_v2_48um_cells': 0.8609, 'S3_20260918_48um_cells': 0.8787, 'S3_20260918_9.6um_pooled': 0.8681},
  'w00': {'HF_v2_48um_cells': 0.9542, 'S3_20260918_48um_cells': 0.9778, 'S3_20260918_9.6um_pooled': 0.9722},
  'consequence_flagged_not_acted_on': 'On 252405 the 2.4 um reference itself scores 0.86-0.88, so the plan\'s 0.85 go bar for a 9 um reader sits 0.01-0.03 below the 2.4 um map there (on w00 the gap is 0.10-0.13). Thresholds are kept exactly as the plan states; whether to rescale them is not the machine\'s call.'},

 'members': {
  'ondisk': {
   'released_s43_75k': {'files': [CK % (43, 75000)], 'sha256': [sha(CK % (43, 75000))], 'licence': 'MIT', 'role': 'baseline and R1 candidate'},
   'soup42': {'files': [CK % (42, s) for s in SOUP_STEPS], 'sha256': [sha(CK % (42, s)) for s in SOUP_STEPS], 'licence': 'MIT', 'role': 'R1 candidate, R4 member'},
   'soup43': {'files': [CK % (43, s) for s in SOUP_STEPS], 'sha256': [sha(CK % (43, s)) for s in SOUP_STEPS], 'licence': 'MIT', 'role': 'R1 candidate, R4 member'},
  },
  'soup_rule': 'uniform arithmetic mean of every floating-point state_dict tensor over steps 40k/50k/60k/75k WITHIN one seed, never across seeds; non-float buffers and the embedded training config are taken from the 75k checkpoint [machine_fill_in for the buffer/config detail]',
  'pending_approval': {
   'nieuwlaar_dense_native': {'file': 'dense_native_016000.safetensors', 'bytes': 272767904, 'source': 'HF Nieuwlaar/ink9um-dense-native (+ train_dense_native.json)', 'licence': 'MIT', 'status': 'PENDING-APPROVAL', 'role': 'R1 candidate, R2/R3/R4 member'},
   'klavis_dense': {'file': 'dense9um-w016excluded-step075000.pth', 'bytes': 138367629, 'source': 'HF domenicor046/ink9um-dense', 'licence': 'MIT per its card', 'status': 'PENDING-APPROVAL', 'role': 'R1 candidate, R2/R3/R4 member'},
   'tauil_ft6k': {'file': 'ink9um_native0139_ft6k.pth', 'bytes': 272897829, 'source': 'GitHub TAUIL-Abd-Elilah/cross-scan-ink-transfer release v1.0', 'licence': 'MIT', 'sha256_prefix_suffix': '4a5258085826...fd21ee71', 'status': 'PENDING-APPROVAL', 'role': 'R1 candidate, R2/R3/R4 member'},
   'scheirer_ft_s42': {'file': 'ink9um_ft_s42_step12000.pth', 'bytes': 138363087, 'source': 'GitHub ShribyrLabs/vesuvius-reports release reader-ft-s42', 'licence': 'checkpoint licence unconfirmed', 'status': 'PENDING-APPROVAL', 'role': 'REFERENCE ARM ONLY: never in R1-R4, never a claim base'},
  },
  'pending_rule': 'A PENDING-APPROVAL member that is not on disk at selection time is absent: it is dropped from every recipe, and no recipe is redefined to replace it. R2 and R3 are then not evaluable. R4 is evaluated on its on-disk part {soup42, soup43} and is labelled R4-ondisk in every output. If a pending member arrives later, selection is re-run once over the full set BEFORE 252405 is rendered, and both selections are reported.',
  'excluded': 'Hecate 9.6: reported, never in a decision on PHerc0841 (trained on fine-scan predictions where both scans exist; PHerc0841 has both).',
 },

 'inference': {
  'input': 'native 9.366 um volume 20250821151531 rendered along the mesh (c0841-9um.py renderer, mesh path an argument): 9.61 um in plane (4 px of the 2.4 um control grid), 41 layers at 9.6 um about the mesh, trilinear; w00big = build-deciding-test/ours-stack.npy',
  'stack_order': 'rev: the 21-layer stack reversed along depth before inference (s21[::-1] in c0841-9um.py). Decided from geometry before scoring (2.4 um control read in rev with the same mesh-normal convention; affine det +1). Forward is a control only.',
  'depth_centres_layers': [-4, -2, 0, 2, 4],
  'depth_centre_definition': 'centre c takes layers [mid+c-10, mid+c+10] (21 layers) of the render, mid = layer index of the mesh',
  'ink9_windows': [[0, 17], [2, 19], [4, 21]],
  'member_map': 'per member: raw model output (no (p-0.25)/0.5 display rescale), mean of the 3 windows per centre, then mean of the 5 centres',
  'tta_rule': "mirror TTA = koine infer.py --tta-mirror (mirrors the equal-length in-plane axes of the (17,128,128) input as compute_equal_length_mirror_axes defines; the 17-layer depth axis is never mirrored). Used ONLY in R3. Everything else runs without TTA.",
  'precision': 'as infer.py does on cuda through polite-run.py; same for every arm',
 },

 'recipes': {
  'R1': 'best single member: among members evaluable at selection time (now: released_s43_75k, soup42, soup43; plus the three MIT community members once on disk; never scheirer_ft_s42), apply the selection rule to single members',
  'R2': 'rank-normalised mean of the MIT community members {nieuwlaar_dense_native, klavis_dense, tauil_ft6k}, each with the five-centre depth mean',
  'R3': 'R2 with mirror TTA on every member',
  'R4': 'rank-normalised mean of R2 members plus soup42 and soup43 (equal weight per member); R4-ondisk = {soup42, soup43} while R2 members are pending',
  'rank_normalisation': 'each member map is replaced by (rank within the scoring mask - 0.5) / n, ties averaged; pixels outside the mask are not used',
 },
 'selection': {
  'frame': 'w00big only (995x1245 at 9.61 um, mask = ours-mask & 2.4 render mask, 40 px border excluded, as auc-in-supervision.py). Selection scores are not results.',
  'rule': 'compute AUC (supervised negatives, HF v2) and hp r at read blur 2 px for every evaluable recipe; keep recipes with AUC >= best AUC - 0.01; select the highest hp r at blur 2.',
  'tie_break': 'hp r equal to 3 decimals -> fewer members -> lower recipe number [machine_fill_in]',
  'second_arm': 'the best single member (R1) is frozen as well, whatever is selected (ensembles softened letters by 19-25% in the ckpt-ens run)',
  'freeze': 'the recipe JSON (selected recipe, R1 member, member files and sha256) is written and hashed before 252405 is rendered',
 },

 'metrics': {
  'auc': {'definition': 'Mann-Whitney AUC on the scoring grid; positives = pixels with pooled ink-label fraction >= 0.75; negatives = pooled ink-label fraction <= 0 AND pooled supervision fraction >= 0.75; inside the render/model mask with a 40 px border excluded',
          'labels_primary': 'HF v2 (inklabels_v2, supervision_mask_v2), the set the 0.731 / 0.762 w00big baselines used',
          'labels_secondary': 'S3 ink-labels 20260918 (reviewed set), reported, never decisive'},
  'hp_r': {'definition': 'Scheirer hp_score.py (ShribyrLabs/vesuvius-reports commit d4e7b78, 02-9um-ink-reader-benchmark/scripts/hp_score.py): Pearson r between read and key after both are high-passed by normalised convolution, hp(x) = x - G(x*m)/max(G(m), 1e-3), Gaussian sigma = 48.0 um / grid pitch (4.995 px at 9.61 um); scored on core = valid & G(valid, 2 sigma) > 0.999 and read > 0',
           'key': 'ScrollPrize 2.4 um map, model 20260417190342, full-resolution tif area-averaged onto the scoring grid (w00: sha256 %s; 252405: sha256 %s). The earlier w00big run used the ds8 jpg; selection uses the tif.' % (
               sha(os.path.join(G, INKW00)) if os.path.exists(os.path.join(G, INKW00)) else 'not fetched', sha(os.path.join(G, INK252))),
           'read_blur_px': [0, 2, 4], 'blur_rule': 'Gaussian blur of the READ only, before the high-pass; the key is never blurred; arms compare only at equal blur',
           'null': '48 shifted keys: (dy, dx) in {-3,-2,-1,0,1,2,3} mm squared minus (0,0), 1 mm = 104 px at 9.61 um, overlap only (no wrap), as SHIFTS in c0841-9um.py; report mean and max; a read clears the null when its zero-shift hp r > the null max'},
  'controls_on_252405': {'forward_order': 'same centres, stack not reversed',
                         'depth_shuffled': '21-layer stack permuted by numpy default_rng(0).permutation(21) before the rev flip [seed machine_fill_in]',
                         'phase_scrambled_tiles': '2D FFT phase replaced by uniform random phase (default_rng(0)), amplitude kept, per tile [machine_fill_in]',
                         'pass': 'forward and depth-shuffled reads have hp r at blur 2 <= their own shifted-key null max'},
  'sanity_gates_before_any_model_on_252405': ['model-free direct render (mean of the 5 layers about the mesh) inside its shifted-key null, by raw Pearson r AND by hp r at blur 0 [hp r half machine_fill_in]; on w00big the raw r was +0.032 against null max +0.066 (vs the 2.4 um Hecate map)',
                                              'local shift scan of released s43_75k against the 2.4 um map peaks within 1 px of (0,0)'],
 },

 'thresholds': {
  'decisive_number': 'AUC (supervised negatives, HF v2) of the frozen selected recipe on 252405',
  'go_to_letter_read': 'AUC >= 0.85 AND hp r at blur 2 > shifted-key null max AND forward and depth-shuffled controls inside their nulls',
  'ambiguous': '0.80 <= AUC < 0.85: read anyway, result marked ambiguous (the public 0.81 readers were not legible). AUC >= 0.85 with a failed hp r or control condition also falls here [machine_fill_in]',
  'exhausted': 'AUC < 0.80: the existing readers have run out; next lever is the pre-registered twin-scan student',
  'experiment_go': 'all of: matched letters inside 252405 >= the bar; precision >= 0.8; zero letters called on null tiles. The released s43_75k matched count is reported beside it.',
  'experiment_no_go': 'AUC < 0.80, or fewer than 5 matched letters',
 },
 'letters': {
  'reader_A': 'sees only the ScrollPrize 2.4 um map of 252405; a fresh agent with no 9 um tiles in context, or me; marks letter boxes and identities; N_ref = count',
  'n_ref_augment': 'if N_ref < 15, add the 552896 blobs after removing every pixel within 40 voxels of w00 points, and recount (plan text, kept verbatim)',
  'bar_rule': '10 matched letters if N_ref >= 14, else 70% of N_ref, rounded UP [rounding machine_fill_in]',
  'match_rule': 'same letter identity AND centre within 0.5 mm (52.0 px at 9.61 um) on the 252405 grid; one-to-one, greedy by distance, each reference letter matched at most once [one-to-one machine_fill_in]',
  'precision': 'matched / all letters called on 252405 tiles',
  'null_rule': 'any letter called on a null tile (phase-scrambled, depth-shuffled) counts against the result; experiment GO needs zero',
  'reader_B_pool': '9 um tiles only, mixed: 252405 tiles, known w00 9 um positives, phase-scrambled nulls, depth-shuffled 252405 tiles; reader calibration on w00big tiles of released s43_75k (floor) and Hecate 9.6 (contaminated ceiling)',
  'estimate_letters_in_252405': 'about 20 (0.2578 cm2 ink-labelled, S3 20260918, against w00 0.3560 cm2 for about 30) [estimate]',
 },
 'leakage': {
  'models': 'no member trained on PHerc0841 or an eligible scroll per its card/config; only MIT checkpoints are claim bases',
  'test_text': '252405 shares zero co-supervised points within 80 voxels of w00 or 552896 (gate c); selection on w00big only, among at most four named recipes',
  'labels_and_maps': 'used only to score; never to pick depths, windows, members or orientation',
 },
 'eligibility_recheck': 're-read prizeEligibility.json before every long run',
}

txt = json.dumps(P, indent=1, ensure_ascii=True) + '\n'
with open(OUT, 'w', encoding='ascii', newline='\n') as f:
    f.write(txt)
h = sha(OUT)
with open(os.path.join(G, 'prereg.sha256'), 'w', newline='\n') as f:
    f.write('%s  prereg.json\n' % h)
print('wrote %s (%d bytes) at %s' % (OUT, len(txt), P['written_local']))
print('sha256 %s' % h)
