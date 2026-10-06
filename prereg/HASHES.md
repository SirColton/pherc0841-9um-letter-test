# Pre-registration hashes

Each file in this folder was frozen before the numbers it governs existed. "Frozen" here means its sha256 was written beside it and committed to my private git history. The order of events is in README.md, under "Pre-registration record".

**Not all copies are identical.**
- Three files are published byte-identical, so `sha256sum` reproduces the frozen hash: `prereg-v2.json`, `prereg-v2-note-2.txt` and `recipe-frozen.json`.
- The other seven are redacted copies, so their hashes cannot match the frozen ones. That is expected.
- The redactions remove:
  - letter identities, replaced by reference ids (L1-2 … L2-7) or "[letter]", because the data terms bar partial transcriptions;
  - local paths and private tooling;
  - my name and the names of internal sessions, replaced by first-person wording ("me", "my"), by "approval" in status labels, or by "a reviewer".
- Nothing else was changed. No rule, number, threshold or timestamp was touched.

## 1. Pre-registration files

| published file | sha256 of the original, as frozen | sha256 of the published copy | status |
|---|---|---|---|
| `day1-prereg-v1.json` | `2d314b1ee767bd52e5b97690877ac9b8e18c535931d2dabe0b46e45dcd99d4cf` | `dbe80c3f8471742ca46f99ab75a6ed22a42f3f68d4672befe854be9e7fa6363d` | redacted |
| `AMENDMENT-1.txt` | `330c64bef6d1b73b71467c3e369a77c3770bd00997950bd6045a3d56bc05be0f` | `d382289aa789dfb7b5069e3620c1db5f782ad5ae4f7c40c9f35e9bd92f27b9a9` | redacted |
| `prereg-v2.json` | `49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c` | `49633d68793a4fef5bbe703bfb67d05d6a906b34d16e58166d0c89664dce401c` | identical |
| `prereg-v2-note-1.txt` | `f657258091915c27961cebe7db31d9089efeb06742dafc82512bdd7d9e8af69c` | `654d291e4ccf0f8465de9241f90f735318f1981f11a7f63b53ea490ad596dc2e` | redacted |
| `prereg-v2-note-2.txt` | `36710c097b26b8314d29decf8627577c13c4c0d3f0031e23f671e44c561e7fc5` | `36710c097b26b8314d29decf8627577c13c4c0d3f0031e23f671e44c561e7fc5` | identical |
| `prereg-v2-note-3.txt` | `6af727dc352fdf26881135981cf58039f3e5fc14f5d3aec9c23aa90805a96454` | `7d0446e96ace4fab0a98366c3f955a0fba9cac3121ae8fc1842d7a3cb0964278` | redacted |
| `prereg-v2-note-4.txt` | `81e5d63a209a1dd56c12485fbb72fd2b0f432446a2c20d6a8cf24c8320f7261b` | `4b19554f8761aa4db9e01f068fde7189fbee5b62b94a72e9a694ac8ecc90d3a6` | redacted |
| `recipe-frozen.json` | `72ad196adb48ce3e356b435f3b087c2b92b95b252186f7e9bbd960f74724e897` | `72ad196adb48ce3e356b435f3b087c2b92b95b252186f7e9bbd960f74724e897` | identical |
| `prereg-student.json` | `f3ebf0abc3f93a6772ce31f6915d7f5aef2ea4afc72badeb11cbb735216d3710` | `3f0c263c81ba0c51607045a537d70d13d7c5da105ffaae55240597c5794dd68e` | redacted |
| `prereg-student-amendment-1.txt` | `3b0f1bc434decb39e9b46e87ef5913c4bd53eb9c1a649a3ccb93f6d69f82cf10` | `a964e5c430d30a24dae66b5ffd1b12ffe4402261de1211d435dad5b2256e84db` | redacted |

**Where each frozen hash is also recorded**, so the files cross-check each other:
- `prereg-v2.json` lists the hashes of `day1-prereg-v1.json` and `AMENDMENT-1.txt` under `inputs_sha256`.
- `recipe-frozen.json` lists the hashes of `prereg-v2.json` and notes 1 to 3.
- `prereg-student.json` lists all of the above, plus note 4.

**What each redaction removed:**

| file | removed |
|---|---|
| `day1-prereg-v1.json` | the local path of the plan file, my name in eight places (two in prose, now "me"; one key name and five status labels, now "approval"), and an internal workflow label in `written_by` |
| `AMENDMENT-1.txt` | the name of a reviewing assistant session, in four places |
| `prereg-v2-note-1.txt` | two letter identities from the w00big calibration, and an internal session name |
| `prereg-v2-note-3.txt` | the identities of the 8 reference letters, and an internal session name |
| `prereg-v2-note-4.txt` | the identities of the 6 letters the ceiling read matched |
| `prereg-student.json` | three letter identities in a summary of day 1, two internal session names, a private folder name and a private tool name |
| `prereg-student-amendment-1.txt` | my name (that sentence now reads in the first person) |

## 2. Code that the pre-registrations pin

`prereg-student.json` (`inputs_sha256`) and the day-1 freeze pin these scripts by hash.

| published file | pinned | original as published from | published copy | status |
|---|---|---|---|---|
| `code/day1-ink9-maps.py` | `bfcb478fe17fcd0a` | `bfcb478fe17fcd0a` | `bfcb478fe17fcd0a` | identical |
| `code/day1-score-baselines.py` | `1faeb063040f574e` | `1faeb063040f574e` | `1faeb063040f574e` | identical |
| `code/day1-select-loaders.py` | `dad9de51c0c86055` | `dad9de51c0c86055` | `dad9de51c0c86055` | identical |
| `code/day1-test-shuffle41.py` | `e1963d7c959a035d` | `e1963d7c959a035d` | `e1963d7c959a035d` | identical |
| `code/student-test-ink9-maps.py` | `bfcb478fe17fcd0a` | `bfcb478fe17fcd0a` | `bfcb478fe17fcd0a` | identical |
| `code/student-test-shuffle41.py` | `e1963d7c959a035d` | `e1963d7c959a035d` | `e1963d7c959a035d` | identical |
| `code/letter-read/note4-verdict.py` | `f4727dc2573830b7` | `f4727dc2573830b7` | `f4727dc2573830b7` | identical |
| `code/day1-labels-hf.py` | `945369b8551884fe` | `945369b8551884fe` | `6ed4c49a7656bdd7` | redacted copy of the pinned version |
| `code/day1-render-9um.py` | `155f35f1ebdbae34` | `155f35f1ebdbae34` | `0634ed749d5c2de4` | redacted copy of the pinned version |
| `code/day1-test-score.py` | `ad38544020e52897` | `ad38544020e52897` | `d03add1d5ddd258d` | redacted copy of the pinned version |
| `code/student-freeze-geom.py` | `03fa32904fcee07d` | `03fa32904fcee07d` | `e19bce23ebdae603` | redacted copy of the pinned version |
| `code/student-train.py` | `632893903340e528` | `632893903340e528` | `f82e3b37a6ec758a` | redacted copy of the pinned version |
| `code/student-test-score.py` | `f798e0d01b0037cb` | `f798e0d01b0037cb` | `5b1f9ae7a5bf8c88` | redacted copy of the pinned version |
| `code/student-test-valstrip.py` | `39b7f4fabaec0637` | `39b7f4fabaec0637` | `3891ab80d5bf17ae` | redacted copy of the pinned version |
| `code/polite-run.py` | `3ec18d3edc72903c` | `3ec18d3edc72903c` | `2f5d9a777d66eda9` | redacted copy of the pinned version |
| `code/runfile.py` | `d5506bcfe8ad2887` | `d5506bcfe8ad2887` | `2dcd47f66717bce1` | redacted copy of the pinned version |
| `code/day1-letter-read-workflow.js` | `bddbcce1de87acb2` | `bddbcce1de87acb2` | `163e9161daa6de2e` | redacted copy of the pinned version |
| `code/letter-read/score.py` | `34daa9f3c50c521a` | `d36b8f83187c8e9d` | `173b0bf8a5a61be7` | later version, redacted |
| `code/letter-read/make-pool.py` | `af4346b15efc8369` | `c20504f60a4c558b` | `40a47797149493aa` | later version, redacted |

The hashes are shown to 16 hex digits. The pinned values are given in full in `prereg-student.json`.
- **The two later versions** add development-subset options that no 252405 pool used.
- **Counts:** the counts in `results/letter-read-252405.csv` come from the score files written at the time of each read, not from a re-run of these published copies.
- **`day1-ink9-maps.py`** gained two command-line flags after the recipe freeze (hash `d6ae6ed3…` became `bfcb478f…`), before any 252405 map was made. With neither flag set, the code path is the frozen one. `prereg-student.json` pins the later hash.

## 3. Unpublished files whose hashes were recorded before use

These files are withheld because they hold letter identities or positions, or are model outputs or data products. Their hashes are listed so that the originals can be checked if they are ever shared.

| file | role | sha256 |
|---|---|---|
| `252405-ref.json` | the locked letter reference (8 scorable letters) | `b12f3b245cb31467f497e5a411905e61df8f9172b5b633e1f9ce57889e9ff4b1` |
| `pd-key.json` | ceiling pool key, recorded before any reader ran | `50502dfb44dd502b66f18bf284877361074f7e108affa07a640808b0f943ec93` |
| `pe-key.json` | arm A pool key | `c3a234fcc956ed74d306b1dac007f5d8cd53e8cd5eb0506069b4e118d2da7422` |
| `pf-key.json` | baseline pool key | `d9e0b30a1f60a64b146a9137023c7e01aa22048f4722d999f718312ff261f93f` |
| `pg-key.json` | student pool key | `cf6b0bb7ccc8da12f42d176524295eb9bb427ad1bffa47fb7a023c321ac6308c` |
| `252405-spmap-grid.npy` | ScrollPrize's 2.4 µm map on the render grid | `75cef7ea157f14e2581050b30b1e782457b8a85fcbae70f1f4247ee920106174` |
| `arm-A-R1-tauil_ft6k-rev-mean5.npy` | ft6k map of 252405 | `5312246bcd3e364480c7832817e9f7c9727e93f44b2d91533af0cb1c54c34337` |
| `baseline-s43_75k-rev-mean5.npy` | released `ink_9um` map of 252405 | `16766c21eab8efbb0d86939506b21a9d0e261c9b7f67fdf2013ace844c7e090c` |
| `student-rev-mean5.npy` | student v1 map of 252405 | `110823285bb403dbb425709964a2136ae5ece2e0fe959473ecaa1f35d8ebde7f` |
| `student-final-step012000.pth` | student v1 weights (not released) | `2a1e01914a2615812a689238e9c92db3484e44f6b7d5f97eac93920041ff8dc0` |
| `scramble-final-step012000.pth` | scramble twin weights (not released) | `e5de283853dcdca95dd5fb3ca0963f0793a790615ad39ff34976c7ca63ae02a1` |

**The limit of all this:** every freeze time comes from my own git history, and no third party timestamped it. Publishing the hashes now makes later edits detectable. It cannot prove when a file was first written.
