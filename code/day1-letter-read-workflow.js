export const meta = {
  name: 'day1-letter-read',
  description: 'Blind letter read of 9 um ink-map tiles: readers name each letter with a box, nulls mixed in, at most one control per batch',
  whenToUse: 'After make-pool.py cut a pool and score.py lock locked the reference; score the written calls with score.py score',
  phases: [
    { title: 'Key', detail: 'one parent agent runs score.py manifest: tile list, which are controls, reference lock; readers never see it' },
    { title: 'Read', detail: 'readers see only their tile PNGs and name letters with boxes, args.concurrency at a time (default 3, max 8)' },
    { title: 'Write', detail: 'one agent writes every call to args.out_path and checks the count' },
  ],
}

// args: {tiles_dir, key_path, reference_path, out_path, per_batch?, concurrency?, panel_px?, pitch_um?,
//        seed?, zoom_dir?, lock_path?, allow_overlap?, python?, score_script?}
// The key and the reference are read only by the parent agent of the Key phase, through score.py manifest,
// which returns names, control flags and source boxes; no reader prompt carries either path or anything
// from them. The reference is only checked against its lock here, so reads can never shape it.
const A = args || {}
for (const k of ['tiles_dir', 'key_path', 'reference_path', 'out_path']) {
  if (typeof A[k] !== 'string' || !A[k]) throw new Error(`args.${k}: a path string is required`)
}
const strip = p => p.replace(/[\\/]+$/, '')
const TILES_DIR = strip(A.tiles_dir)
const SEP = TILES_DIR.includes('\\') ? '\\' : '/'
const PY = A.python || 'python'
const SCORE = A.score_script || 'letter-read\\score.py'
const ZOOM = strip(A.zoom_dir || TILES_DIR + '-zoom')
const OUT = A.out_path
const CONC = Math.max(1, Math.min(8, Math.floor(A.concurrency || 3)))
const PER = Math.max(1, Math.floor(A.per_batch || 2))
const SEED = A.seed === undefined ? 4211 : A.seed
const ALLOW_OVERLAP = A.allow_overlap === true
const lc = p => p.replace(/\//g, '\\').toLowerCase()
if (lc(ZOOM) === lc(TILES_DIR) || lc(ZOOM).startsWith(lc(TILES_DIR) + '\\')) throw new Error('args.zoom_dir must be outside the tiles folder')
for (const k of ['key_path', 'reference_path', 'out_path']) {
  if (lc(A[k]).startsWith(lc(TILES_DIR) + '\\')) throw new Error(`args.${k} is inside the tiles folder, where readers are pointed`)
}
if ((A.concurrency || 0) > 8) log(`args.concurrency ${A.concurrency} capped at 8`)
log(`tiles ${TILES_DIR}; ${PER} tile(s) per reader, ${CONC} readers at a time; seed ${SEED}; overlapping tiles ${ALLOW_OVERLAP ? 'may' : 'never'} share a batch`)

const GREEK = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu',
  'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']

// ---- Key: the parent reads the key, the readers never do --------------------------------------
phase('Key')
const MANIFEST = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    exit_code: { type: 'number' },
    problems: { type: 'array', items: { type: 'string' } },
    pitch_um: { type: ['number', 'null'] },
    tiles: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          tile: { type: 'string' },
          control: { type: 'boolean' },
          box: { type: 'array', items: { type: 'number' } },
          png_px: { type: 'array', items: { type: 'number' } },
        },
        required: ['tile', 'control', 'box', 'png_px'],
      },
    },
  },
  required: ['ok', 'exit_code', 'problems', 'tiles'],
}
const manifestCmd = `"${PY}" "${SCORE}" manifest --key "${A.key_path}" --tiles-dir "${TILES_DIR}" --reference "${A.reference_path}"` +
  (A.lock_path ? ` --lock "${A.lock_path}"` : '')
const man = await agent(
  `You are the PARENT of a blind letter read, not a reader. Run exactly this one command with the Bash tool (it prints one line of JSON) and nothing else:\n${manifestCmd}\n` +
  `Return that JSON as your structured answer, field for field and tile for tile, with exit_code = the command's exit code. ` +
  `Do not open the key, the reference or any tile yourself, do not summarise, do not drop, add or reorder tiles. ` +
  `If the command cannot run at all, return ok=false, exit_code=-1, the error text in problems and tiles=[].`,
  { label: 'key:manifest', phase: 'Key', schema: MANIFEST })
if (!man || !man.ok || man.exit_code !== 0 || !man.tiles.length) {
  const why = man ? (man.problems.join(' | ') || `exit ${man.exit_code}, ${man.tiles.length} tiles`) : 'the manifest agent returned nothing'
  log(`STOPPED before any read: ${why}`)
  return { error: why, manifest: man, calls: [] }
}
const ENTRIES = man.tiles.map(t => ({ ...t }))
const NAMES = ENTRIES.map(t => t.tile)
if (new Set(NAMES).size !== NAMES.length || NAMES.some(n => !/^[A-Za-z0-9_-]+$/.test(n))) throw new Error(`manifest tile names are not unique opaque ids: ${NAMES.join(' ')}`)
if (ENTRIES.some(t => t.box.length !== 4 || t.png_px.length !== 2)) throw new Error('manifest boxes or png sizes are malformed')
const BY = new Map(ENTRIES.map(t => [t.tile, t]))
let PITCH = man.pitch_um || A.pitch_um || 9.612
if (A.pitch_um && man.pitch_um && Math.abs(A.pitch_um - man.pitch_um) > 1e-6) log(`args.pitch_um ${A.pitch_um} differs from the key's ${man.pitch_um}: the key's is used`)
if (A.panel_px) {
  const off = ENTRIES.filter(t => t.png_px.some(v => v !== A.panel_px))
  if (off.length) log(`args.panel_px ${A.panel_px} differs from ${off.length} tiles' PNG size (e.g. ${off[0].png_px.join('x')}); the PNG sizes are used`)
}
const nCtl = ENTRIES.filter(t => t.control).length
log(`manifest: ${ENTRIES.length} tiles (${ENTRIES.length - nCtl} real, ${nCtl} controls), pitch ${PITCH} um, reference lock OK`)

// ---- batching: seeded, at most one control per batch, never two tiles from overlapping places -----
// Math.random is not allowed in a workflow (it breaks resume), so a seeded mulberry32: same tiles +
// same seed -> same batches.
function rand(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const rnd = rand(SEED)
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}
// Two tiles whose source windows share more than 0.5 mm both ways would let a reader compare them: two
// real tiles corroborate each other, and a null beside its own source window shows its blocks. Neither
// may share a batch unless args.allow_overlap.
const TOL = 500 / PITCH
const overlaps = (a, b) => Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0]) > TOL &&
  Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]) > TOL
function makeBatches(names) {
  const out = []
  for (const t of shuffle(names.map(n => BY.get(n)))) {
    const b = out.find(b => b.length < PER && !(t.control && b.some(x => x.control)) && (ALLOW_OVERLAP || !b.some(x => overlaps(x, t))))
    if (b) b.push(t)
    else out.push([t])
  }
  for (const b of out) shuffle(b)
  const twoCtl = out.filter(b => b.filter(t => t.control).length > 1)
  const ov = ALLOW_OVERLAP ? [] : out.filter(b => b.some((x, i) => b.some((y, k) => k > i && overlaps(x, y))))
  if (twoCtl.length || ov.length || out.flat().length !== names.length || out.some(b => b.length > PER)) throw new Error(`batching broke: ${JSON.stringify(out.map(b => b.map(t => t.tile)))}`)
  return out.map(b => b.map(t => t.tile))
}

// ---- the reader ---------------------------------------------------------------------------------
const tilePath = n => TILES_DIR + SEP + n + '.png'
function tileLine(n) {
  const t = BY.get(n)
  const um = PITCH * (t.box[2] - t.box[0]) / t.png_px[0]
  const mmH = (t.png_px[0] * um / 1000).toFixed(1)
  const mmW = (t.png_px[1] * um / 1000).toFixed(1)
  const px = mm => Math.round(mm * 1000 / um)
  return `  ${tilePath(n)} : ${t.png_px[1]} px wide x ${t.png_px[0]} px tall = ${mmW} x ${mmH} mm; 1 px = ${um.toFixed(3)} um, so 1 mm = ${px(1)} px and a 2-4 mm letter is about ${px(2)}-${px(4)} px tall`
}
const CONTEXT = `You are reading ink-detection maps of a carbonised Herculaneum papyrus scroll, looking for written Greek letters.
Each tile is ONE grayscale PNG: the output of a machine-learning ink-detection model run on a 9 um X-ray CT scan, flattened onto the papyrus surface. Bright = the model thinks ink is there; dark = no ink. There is no CT image, no second panel and no label: the map is all there is.
Greek letters in these scrolls are typically 2-4 mm tall and sit in horizontal rows roughly 4-6 mm apart. The model does not draw a pen stroke: strokes can come out as soft, blurred bands, thinner or much wider than a pen, and neighbouring letters can merge, so judge shape and spacing rather than stroke width. Text can appear mirrored (seen from behind) or rotated: if so, name the letter as it reads upright and unmirrored, and say so in notes.
Black areas, if any, are outside the scanned surface. Some tiles may hold no writing at all; judge every tile on its own evidence. Missing a real letter and naming a letter that is not there are equally bad.
You may look at ONLY the tile PNG(s) named below. Do NOT open, list or search any other file or folder: no other tile, no key, no label, no reference, no other map or render, nothing in any parent folder, and use no network.
You may zoom: run Python with ${PY} (numpy and PIL are installed) on your own tiles to crop, enlarge, adjust contrast, or draw a labelled grid of TILE coordinates (a line and its value every 50 px) so you can read positions; save crops only under ${ZOOM}${SEP}<tile name>${SEP}<round>${SEP} (create it) and Read them. Write nowhere else.`

const readSchema = names => ({
  type: 'object',
  properties: {
    tiles: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          tile: { type: 'string', enum: names },
          letters: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                letter: { type: 'string', enum: GREEK },
                box: { type: 'array', items: { type: 'number' }, minItems: 4, maxItems: 4, description: '[x0, y0, x1, y1] in the tile PNG\'s own full-resolution pixels: x = column from the left edge, y = row from the top, 0-based' },
                confidence: { type: 'number' },
                note: { type: 'string' },
              },
              required: ['letter', 'box', 'confidence'],
            },
          },
          zoom_files: { type: 'array', items: { type: 'string' }, description: 'every crop you saved and Read for this tile' },
          notes: { type: 'string' },
        },
        required: ['tile', 'letters', 'zoom_files'],
      },
    },
  },
  required: ['tiles'],
})

const readPrompt = (b, round) => `${CONTEXT}\n\nRound: ${round}. Your tile(s):\n${b.map(tileLine).join('\n')}\n\n` +
  `For EACH tile: Read it, then zoom before you name anything, because the first view can be scaled down: crop it in pieces at full resolution or larger, with a labelled coordinate grid, save the crops under ${ZOOM}${SEP}<tile>${SEP}${round}${SEP} and Read every crop; list them in zoom_files. ` +
  `Then name every letter you can identify. For each: letter = its lowercase English name from the list (a C-shaped lunate sigma is "sigma"); box = [x0, y0, x1, y1] around that one letter in the TILE's own pixel coordinates at full resolution (convert back from any crop's offset and enlargement; never report crop or scaled-view coordinates); confidence 0-1. ` +
  `One call per letter; never name two letters at the same spot. Strokes or rows you cannot name as a specific letter are not calls: describe them in notes. An empty letters list is a valid answer.`

async function waves(items, n, fn) {
  const out = []
  for (let i = 0; i < items.length; i += n) {
    const part = await parallel(items.slice(i, i + n).map((it, k) => () => fn(it, i + k)))
    out.push(...part)
  }
  return out
}

const zoomed = (t, round) => (t.zoom_files || []).some(f => new RegExp(`[\\\\/]${t.tile}[\\\\/]${round}[\\\\/][^\\\\/]+$`).test(f))
function clean(t, round) {
  const t0 = BY.get(t.tile)
  const problems = []
  const letters = (t.letters || []).filter(L => {
    const ok = Array.isArray(L.box) && L.box.length === 4 && L.box.every(v => typeof v === 'number' && isFinite(v))
    if (!ok) problems.push(`dropped a call with a malformed box: ${JSON.stringify(L)}`)
    return ok
  }).map(L => {
    const box = L.box.map(v => Math.round(v))
    const cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2
    if (cx < 0 || cy < 0 || cx > t0.png_px[1] || cy > t0.png_px[0]) problems.push(`${L.letter} box ${box.join(',')} has its centre outside the ${t0.png_px[1]}x${t0.png_px[0]} tile`)
    return { letter: L.letter, box, confidence: Math.round((L.confidence || 0) * 100) / 100, note: L.note || '' }
  })
  return { tile: t.tile, round, letters, zoom_files: t.zoom_files || [], zoomed: zoomed(t, round), notes: t.notes || '', problems }
}

async function readAll(names, round) {
  const batches = makeBatches(names)
  batches.forEach((b, i) => log(`${round} batch ${i + 1}/${batches.length}: ${b.join(' ')} | control: ${b.filter(n => BY.get(n).control).join(' ') || 'none'}`))
  const reads = await waves(batches, CONC, b => agent(readPrompt(b, round), { label: `${round}:${b.join(',')}`, phase: 'Read', schema: readSchema(b) }))
  const got = new Map()
  reads.forEach((r, i) => ((r && r.tiles) || []).forEach(t => {
    if (!batches[i].includes(t.tile)) return
    if (got.has(t.tile)) log(`${round}: ${t.tile} answered twice in one batch; the first answer stands`)
    else got.set(t.tile, clean(t, round))
  }))
  return { batches, got }
}

// ---- Read ---------------------------------------------------------------------------------------
phase('Read')
const first = await readAll(NAMES, 'read')
// One re-read, fresh agents, for any tile with no answer or no crop under its own round folder.
const redo = NAMES.filter(n => !first.got.has(n) || !first.got.get(n).zoomed)
let second = { batches: [], got: new Map() }
if (redo.length) {
  log(`re-reading ${redo.length} tiles (no answer, or no crop listed under <tile>${SEP}read${SEP}): ${redo.join(' ')}`)
  second = await readAll(redo, 'reread')
}
// The same rule for every tile, real or control: a zoomed read beats an unzoomed one, else the first
// read stands; the other read rides along in reads[], never in calls[].
const final = new Map()
for (const n of NAMES) {
  const a = first.got.get(n), b = second.got.get(n)
  if (a && b) final.set(n, (b.zoomed && !a.zoomed) ? { ...b, other: a } : { ...a, other: b })
  else if (a || b) final.set(n, a || b)
}
const noRead = NAMES.filter(n => !final.has(n))
const unzoomed = NAMES.filter(n => final.has(n) && !final.get(n).zoomed)
if (noRead.length) log(`NO READ after the re-read: ${noRead.join(' ')} (score.py refuses until they are read, or --allow-unread)`)
if (unzoomed.length) log(`no crop listed after the re-read: ${unzoomed.join(' ')}`)
const sorted = [...final.keys()].sort()
const calls = []
for (const n of sorted) for (const L of final.get(n).letters) calls.push({ tile: n, letter: L.letter, box: L.box, confidence: L.confidence, round: final.get(n).round })
const boxProblems = sorted.flatMap(n => final.get(n).problems.map(p => `${n}: ${p}`))
if (boxProblems.length) log(`call problems (kept in the file, score.py judges them): ${boxProblems.join(' | ')}`)
log(`${sorted.length} of ${NAMES.length} tiles read; ${calls.length} letters called`)

// ---- Write: the script has no filesystem, so one agent writes the file and score.py count checks it --
phase('Write')
const reads = sorted.map(n => {
  const r = final.get(n)
  return { tile: n, round: r.round, n_letters: r.letters.length, n_zoom: r.zoom_files.length, zoomed: r.zoomed, notes: r.notes.slice(0, 400) }
})
const head = {
  schema: 'day1-letter-read/1', tiles_dir: TILES_DIR, seed: SEED, per_batch: PER, concurrency: CONC, pitch_um: PITCH,
  tiles: NAMES.length, tiles_read: sorted, no_read: noRead, unzoomed,
  batches: { read: first.batches, reread: second.batches },
}
const text = ['{']
  .concat(Object.entries(head).map(([k, v]) => ` ${JSON.stringify(k)}: ${JSON.stringify(v)},`))
  .concat([' "reads": [', reads.map(r => '  ' + JSON.stringify(r)).join(',\n'), ' ],'])
  .concat([' "calls": [', calls.map(c => '  ' + JSON.stringify(c)).join(',\n'), ' ]', '}'])
  .join('\n')
const want = { n_calls: calls.length, box_sum: calls.reduce((s, c) => s + c.box.reduce((u, v) => u + v, 0), 0), n_reads: reads.length }
const WROTE = {
  type: 'object',
  properties: { written: { type: 'boolean' }, n_calls: { type: 'number' }, box_sum: { type: 'number' }, n_reads: { type: 'number' }, problems: { type: 'string' } },
  required: ['written', 'n_calls', 'box_sum', 'n_reads'],
}
const writePrompt = `You write one file; you are not a reader and you judge nothing. Write the JSON text between the two marker lines to\n${OUT}\nEXACTLY as given (Write tool; create its folder if needed; no reformatting, nothing added or dropped; touch no other file). ` +
  `Then run this one command with the Bash tool and return the three numbers it prints:\n"${PY}" "${SCORE}" count --reads "${OUT}"\n-----BEGIN-----\n${text}\n-----END-----`
let check = null
for (let attempt = 1; attempt <= 2; attempt++) {
  check = await agent(writePrompt, { label: `write:${attempt}`, phase: 'Write', schema: WROTE })
  if (check && check.written && check.n_calls === want.n_calls && check.box_sum === want.box_sum && check.n_reads === want.n_reads) break
  log(`write ${attempt}: the file does not check (${JSON.stringify(check)} vs ${JSON.stringify(want)})${attempt === 1 ? '; writing again' : ''}`)
  check = check ? { ...check, mismatch: true } : null
}
const writeOk = !!(check && !check.mismatch)
log(writeOk ? `wrote ${OUT}: ${want.n_calls} calls on ${want.n_reads} tiles, checked` : `${OUT} NOT verified: rebuild it from this run's return value`)
return {
  out_path: OUT, write_ok: writeOk, write_check: check, expected: want,
  tiles: NAMES.length, tiles_read: sorted, no_read: noRead, unzoomed, call_problems: boxProblems,
  calls, reads: sorted.map(n => ({ ...final.get(n) })), batches: { read: first.batches, reread: second.batches },
}
