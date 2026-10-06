# Reproducing this work

This repository holds my code, my pre-registrations (redacted where the data terms require it) and counts-only results. It holds no scan data, ink maps, label images, renders, tiles, checkpoints or letter identities. Everything below is fetched from its public source.

## 1. Environment

The environment I ran in, with versions as installed when I staged this repository:

| component | version |
|---|---|
| Python | 3.14.6 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| torch | 2.13.0+cu126 |
| tifffile | 2026.8.23 |
| zarr | 3.3.0 |
| Pillow | 12.3.0 |
| Node.js | 24 (only for the workflow mock test) |

You also need:
- **ScrollPrize/villa's ink-detection package `koine_machines`** on `PYTHONPATH`, from its `merge-ink-pipelines` branch as fetched during this work. My map runs call its `inference/infer.py` unchanged. That file's sha256 was `b565fca5f6ef5aaed0add25a1f19212a31a73efb7c5a8a79a837d9ba0a24ce2f`, as `prereg/prereg-student.json` records under `inputs_sha256`. I did not record a villa commit hash. That is a gap.
- **villa's `vesuvius` package**, for `normalize_robust`. Only the student training uses it.
- **`safetensors`**, to read the ink9um-dense-native checkpoint.
- **A GPU** for the map runs and the student training. One 8 GB card was enough: bf16 autocast, batch 8.

Every GPU job ran through `code/polite-run.py`, which lowers the process priority and caps CPU threads at four.

## 2. Layout

The scripts expect the layout they were written in. `<workdir>` is one directory:
- `<workdir>/*.py` and `*.js`: everything in `code/` except `code/letter-read/`.
- `<workdir>/geom/day1-9um/letter-read/`: everything in `code/letter-read/`.
- `<workdir>/geom/...`: data the scripts fetch or write. Each script's docstring names the folders it reads and writes.

Run every script from `<workdir>`. Paths inside the scripts are relative to it.

## 3. Data (all public)

- **Bucket:** `https://vesuvius-challenge-open-data.s3.amazonaws.com/`
- **9.366 µm volume:** `PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr/0/`
- **2.403 µm volume:** `PHerc0841/volumes/20260319124803-2.403um-0.2m-77keV-masked.zarr/`
- **Segments** (meshes and ScrollPrize's 2.4 µm ink maps, under `PHerc0841/segments/<segment>/`):

  | short name | segment folder |
  |---|---|
  | w00 | `20260220213127-w00` |
  | 552896 | `20260220214732-auto_grown_20260220144552896` |
  | 252405 | `20260221022814-auto_grown_20260220174252405` |

- **Ink maps:** `<segment>/ink-detection/PHerc0841-<segment id>-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif`
- **Labels:** `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/<segment>/`, with HF v2 ink labels, supervision masks and validation masks. `code/day1-labels-hf.py` fetches them and carries them onto the render grid.
- **Eligibility list, re-read before the test:** `https://raw.githubusercontent.com/ScrollPrize/villa/main/scrollprize.org/src/data/prizeEligibility.json`

The scan data are CC BY-NC 4.0. Cite them as the data page's "How to Cite" section asks.

## 4. Checkpoints (linked, never rehosted)

| role | source | file | bytes | sha256 | licence as stated at the source |
|---|---|---|---|---|---|
| frozen recipe (arm A), student base | GitHub TAUIL-Abd-Elilah/cross-scan-ink-transfer, release v1.0 | `ink9um_native0139_ft6k.pth` | 272,897,829 | `4a5258085826f4744dbbaff3ec86ea0dd6c565f0d40ca706c7b62e5fbd21ee71` | repository MIT; README: data CC BY-NC 4.0 |
| baseline, soups | Hugging Face `scrollprize/ink_9um` | the released seed 43, step 75,000 (my local name: `seed43-step-075000.pth`), plus seeds 42 and 43 at steps 40k to 75k for the soups | — | see `prereg/day1-prereg-v1.json` | MIT (metadata) |
| selection only | Hugging Face `Nieuwlaar/ink9um-dense-native` | `dense_native_016000.safetensors` | 272,767,904 | `d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d` | MIT (metadata) |
| selection only | Hugging Face `domenicor046/ink9um-dense` | `dense9um-w016excluded-step075000.pth` | 138,367,629 | `1d7b99bc641b459736a762b2a1d8c357f61ada4f0117c24c7327b6a1293ba1eb` | metadata MIT; the card says the checkpoints remain subject to their sources' terms |
| reference arm only | GitHub ShribyrLabs/vesuvius-reports, release reader-ft-s42 | `ink9um_ft_s42_step12000.pth` | 138,363,087 | `aed5246eeba41a840acb12073f152a41725843a7e6bab378856365c640f08020` | none stated for the checkpoint |

`code/day1-select-loaders.py` reads each checkpoint with `weights_only=True` or `safetensors`, never with pickle. It rewrites each one into the payload `infer.py` expects, and proves a strict load.

Labels used in the code and in the pre-registrations:
- "scheirer" / "Scheirer ft_s42" means the ShribyrLabs reader-ft-s42 checkpoint.
- "klavis" / "KLAVIS" means domenicor046's ink9um-dense.
- The pre-registration's "MIT per card" for KLAVIS overstates the card. Its own provenance sentence is the one quoted above.

## 5. Steps, in order

Each script's docstring holds its full usage, and the pre-registration files record the exact settings.

1. **Registration.** Run `python fl-register-0841.py fl-register-0841-run.txt`. It writes `reg-pherc0841-affine.json`, which is also in `results/`.
2. **9 µm renders.** Run `python day1-render-9um.py --mesh <segment tifxyz dir> --crop r0 r1 c0 c1 --out <dir> --name <name> --canvas spmap=<2.4 µm ink map> [--canvas inklabels=... --canvas supervision=...] --map spmap`. For 252405 the crop is mesh cells rows 42-468 and cols 210-664. The w00 and 552896 calls are written out in `prereg/prereg-student.json` under `2_training_regions.renders`.
3. **Labels on the render grid.** Run `python day1-labels-hf.py <out dir>`.
4. **Checkpoints into loader payloads.** Run `python day1-select-loaders.py`, then `python day1-soup.py` for the seed soups.
5. **Maps.** Run `python polite-run.py day1-ink9-maps.py <outdir> <name=ckpt> [...] [--orders rev,fwd] [--stack <npy>] [--shuffle21 <seed>]`.
   - It uses the reversed depth order and depth centres −4, −2, 0, +2 and +4.
   - At each centre it averages `infer.py` windows 0-17, 2-19 and 4-21.
6. **Selection on w00big.**
   - Run `python day1-score-baselines.py <maps dir> <out prefix> --sigma 192` (and with the default σ 48).
   - Then run `day1-select-recipes.py` and `day1-select-table.py`.
   - Then run `day1-freeze.py`, which writes the frozen recipe.
7. **The 252405 test.**
   - Run `python day1-test-shuffle41.py <out dir>`.
   - Run the maps from step 5 with `--stack` and `--shuffle21 0` for the controls.
   - Run `python day1-test-score.py <out.txt> <maps dirs...>`, `python day1-gate1-full.py <out.txt>` and `python polite-run.py day1-render-gate2.py <render dir> <name> [<baselines dir>]`.
8. **The letter read.**
   - Lock the reference: `python geom/day1-9um/letter-read/score.py lock --reference REF.json`.
   - Cut the pool: `python geom/day1-9um/letter-read/make-pool.py --map MAP.npy --mask MASK.npy --grid GRID.json --tiles-dir TILES --key KEY.json --seed 24 [--depth-map DEPTHNULL.npy]`. Mirror phase nulls are the default.
   - Run the blind readers with `day1-letter-read-workflow.js`, a script for Claude Code's workflow runner. Its args are `{tiles_dir, key_path, reference_path, out_path, per_batch: 1, python, score_script}`.
   - Score: `python geom/day1-9um/letter-read/score.py score --reads OUT.json --key KEY.json --reference REF.json --out REPORT.json`.
   - Compute the verdict: `python geom/day1-9um/letter-read/note4-verdict.py --pool <pool> [...] --ceiling <pool> --out VERDICT.json`.
9. **The student.**
   - Run `python student-freeze-geom.py`.
   - Render: `python student-render-9um.py ...` (the same interface as step 2), then `python student-renders-check.py` and `python student-renders-labels.py <w00|552896>`.
   - Train: `python polite-run.py student-train.py --out geom/student-9um/train-student`, and again with `--scramble` for the twin.
   - Test: `python polite-run.py student-test-ink9-maps.py ...`, then `python student-test-score.py <out.txt> <maps dirs...>`.
   - Run the letter read as in step 8.
10. **Self-tests**, on synthetic data (102 checks). Run `python code/letter-read/test_letter_read.py` in this repository, or `python geom/day1-9um/letter-read/test_letter_read.py` in the `<workdir>` layout.
    - Node.js is needed for the workflow mock.
    - One check reads the w00 render grid (`geom/student-9um/renders/w00-spmap-grid.npy`), so it fails without that data.
    - On the published copy with no data present, 101 of 102 pass.

**What cannot be reproduced from this repository alone:**
- The 252405 letter reference, the tile pools and the reader calls hold letter identities and positions, so they are not published (data terms). To repeat the letter read, make a reference the same way: a blind reader marks the 2.4 µm map, using `day1-refread-views.py`, `day1-refread-mark.py` and `ref-from-letters.py`.
- The readers are AI agents, so fresh readers will not repeat the calls exactly.

## 6. How the published code differs from what ran

- **35 of the 65 code files are byte-identical** to what ran. The pre-registrations pin some of them by hash, and these can be checked directly (see `prereg/HASHES.md`):
  - `day1-ink9-maps.py`, `day1-score-baselines.py`, `day1-select-loaders.py` and `day1-test-shuffle41.py`;
  - the student copies of the map and shuffle scripts;
  - `note4-verdict.py`.
- **The other 30 differ only in the following ways:**
  - **Absolute local paths** became relative paths. The interpreter default became `python`, and the chunk-cache default became `scroll-data/chunkcache`.
  - **The private load probe** in `student-train-w00bigproof.py` became a `LOAD_LINE` environment variable. The usage lines that mentioned the probe now say "a local load check".
  - **Host names:** `runfile.py` no longer writes the host name into its run banners.
  - **`runfile.py`'s docstring** describes the stale-file incident in general terms, and no longer names unpublished scripts or a private reader tool.
  - **Names in comments and strings:** sentences that named me now read in the first person, status labels that carried my name now read "PENDING-APPROVAL", and internal session names became "a reviewer".
  - **Letter identities:** two comments that named letters seen during the work no longer do. `blob-scale.py` prints reference ids, never letters.
  - **The workflow mock test** uses a fake `X:` drive. It finds the workflow script in either layout.
  - **"Run from …" lines** now say `<workdir>`.
- **Two letter-read files are later versions than the ones the student pre-registration pins.** They are `score.py` and `make-pool.py`.
  - The later versions add options for a development subset, which no 252405 pool used.
  - The counts in `results/letter-read-252405.csv` come from the score files written at the time of each read, not from a re-run of these published copies.
  - The pools themselves were cut by the pinned `make-pool.py` (sha256 `af4346b1…`).
- **`code/student2-*.py`** is the student v2 harness, still in progress when this was staged. Its frozen hash will be the one its pre-registration pins.
- **Where my plan was silent, the AI assistant fixed a detail.** The pre-registration files and docstrings mark those details "[machine]" or "machine_fill_in". The plan file itself (`design.json`) is not published.
