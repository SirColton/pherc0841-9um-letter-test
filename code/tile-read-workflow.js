export const meta = {
  name: 'tile-read',
  description: 'Read scroll render tiles for letters, blind controls mixed in, adversarially verify any find',
  phases: [{ title: 'Read', detail: '8 readers at a time, 4 tiles each, at most one control per batch' }, { title: 'Verify', detail: '3 sceptic lenses per flagged tile (skipped when args.verify is false)' }],
}

const TILES = args.tiles
// args.dir / pitch_um / panel_px: the 9.36 um sweeps ran on the defaults; the 2.4 um sweep passes
// geom\\l0-1203-render, 9.6 and 900 (fl-tiles-l0.py downsamples 4x)
const BASE = '.\\' + (args.dir || 'geom\\grown1203f-render')
const DIR = BASE + '\\tiles'
const PY = args.python || 'python'
const SCRATCH = BASE + '\\review'
const PITCH = args.pitch_um || 9.36
const PANEL = args.panel_px || 1200
const PER = args.per_batch || 4
const VERIFY = args.verify !== false
const MM = (PANEL * PITCH / 1000).toFixed(1)
const px = mm => Math.round(mm * 1000 / PITCH)
// args.controls: the control tile names, read from the key by the parent. Required, because a batch
// holding two controls let them be compared (the 10-05 audit); agents never see the key.
if (!Array.isArray(args.controls)) throw new Error('args.controls: pass the control tile names from the key ([] if none)')
const CONTROLS = new Set(args.controls)
// args.model_strokes: what the model draws, which is not what the pen drew. On the 2.4 um route
// Hecate paints PHerc0841's known letters as bands 1.2-1.7 mm wide, and the first calibration's
// sceptics refuted both positives as "2-3x a normal stroke"; a pen-width rule rejects real ink.
// No letter is named here: the old note named two letters, and the positive was read as those two.
const STROKES = args.model_strokes ? `
  IMPORTANT, measured on this exact model and route over KNOWN text: the model does not draw pen-width strokes. It paints ink as soft, blobby bands about ${args.model_strokes} mm wide, so letters look like thick, blurred, rounded shapes roughly 2.5-4 mm tall, often merged with a neighbour. Judge SHAPE and SPACING, not stroke width: a band 1-2 mm wide is normal here. Only one of the two orders usually shows the letters; the other often shows unrelated scattered blobs, and that disagreement is normal, not a refutation.` : ''
const PEN = args.model_strokes ? '' : `, strokes 0.3-0.6 mm wide (${px(0.3)}-${px(0.6)} px)`

const CONTEXT = `You are reading X-ray CT renders of a carbonised Herculaneum papyrus scroll, looking for written Greek letters.
Each tile is one PNG triptych, about ${3 * PANEL + 20} x ${PANEL + 22} px: three panels of the SAME ${MM} mm x ${MM} mm patch of papyrus surface, side by side.
  Panel 1, "direct render": the CT intensity on the papyrus surface. Horizontal and vertical papyrus fibres usually show as fine stripes. Ink is usually NOT visible in it; when it is, it can show as a slightly brighter or differently textured patch ("crackle"). A blank direct render is no evidence against ink.
  Panel 2, "model, order A": an ink-detection model's output. Bright = the model thinks ink is there.
  Panel 3, "model, order B": the same model with the layer stack fed in the opposite order. One order is usually the right one.
Scale: 1 px = ${PITCH} micrometres; a tile panel is ~${PANEL} px = ${MM} mm. Greek letters in these scrolls are typically 2-4 mm tall (about ${px(2)}-${px(4)} px)${PEN}, and they sit in horizontal rows that run parallel to the horizontal papyrus fibres, rows roughly 4-6 mm apart. Text can appear mirrored (seen from behind) or rotated, so judge shapes in any orientation.${STROKES}
Black areas are outside the surface: stair-stepped where the surface ends, or a straight band along the bottom or right edge where the patch was shorter than a panel. The model often lights up along those borders - that is an edge artifact, not ink.
Some tiles may come from scrolls where text is already known, and some from places with no text at all; judge every tile on its own evidence. Do NOT open any file other than the tile PNGs named to you and your own crops. In particular never read anything named tiles-key.
You may zoom: run Python with ${PY} (PIL is installed) to crop or enlarge regions of a tile and save the crop under ${SCRATCH}\\<your tile name>\\ (create it), then Read the crop. Do not write anywhere else, do not touch any other file, use no network.`

const readSchema = names => ({
  type: 'object',
  properties: {
    tiles: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          tile: { type: 'string', enum: names },
          verdict: { type: 'string', enum: ['letters', 'strokes-not-letters', 'nothing', 'cannot-judge'] },
          confidence: { type: 'number' },
          which_panels: { type: 'string' },
          where: { type: 'string', description: 'approximate box in panel pixels (x0,y0,x1,y1) of anything flagged' },
          letters_guess: { type: 'string', description: 'if letters: which Greek letters you think, else empty' },
          zoom_files: { type: 'array', items: { type: 'string' }, description: 'every crop you saved and Read for this tile; at least one A-*.png and one B-*.png' },
          reasons: { type: 'string' },
        },
        required: ['tile', 'verdict', 'confidence', 'zoom_files', 'reasons'],
      },
    },
  },
  required: ['tiles'],
})

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    still_letters: { type: 'boolean', description: 'true only if after your attack you still see at least one legible letter' },
    count_legible: { type: 'number' },
    reasons: { type: 'string' },
  },
  required: ['refuted', 'still_letters', 'reasons'],
}

async function waves(items, n, fn) {
  const out = []
  for (let i = 0; i < items.length; i += n) {
    const part = await parallel(items.slice(i, i + n).map((it, k) => () => fn(it, i + k)))
    out.push(...part)
  }
  return out
}

// Math.random is not allowed in a workflow (it breaks resume), so batching draws from a seeded
// mulberry32: same tiles + same args.seed -> same batches.
function rand(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const rnd = rand(args.seed === undefined ? 2403 : args.seed)
function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

// Controls go to distinct batches picked at random, sweep tiles fill the rest in the given order,
// and each batch is shuffled so a control's place in it varies.
function makeBatches(names) {
  const ctl = names.filter(t => CONTROLS.has(t))
  const swp = names.filter(t => !CONTROLS.has(t))
  const nb = Math.max(Math.ceil(names.length / PER), ctl.length)
  const cap = Array.from({ length: nb }, (_, i) => Math.floor(names.length / nb) + (i < names.length % nb ? 1 : 0))
  const out = Array.from({ length: nb }, () => [])
  shuffle([...Array(nb).keys()]).slice(0, ctl.length).forEach((b, i) => out[b].push(ctl[i]))
  let k = 0
  for (let b = 0; b < nb; b++) while (out[b].length < cap[b] && k < swp.length) out[b].push(swp[k++])
  for (const b of out) shuffle(b)
  const bad = out.filter(b => b.filter(t => CONTROLS.has(t)).length > 1)
  if (bad.length || k !== swp.length || out.flat().length !== names.length) throw new Error(`batching broke: ${JSON.stringify(out)}`)
  return out.filter(b => b.length)
}

// Each round zooms into the tile's own round folder, so a reader cannot pass the check on another
// tile's crops or on the first reader's.
const zoomedBoth = t => ['A', 'B'].every(p => (t.zoom_files || []).some(f => new RegExp(`[\\\\/]${t.tile}[\\\\/]${t.round}[\\\\/]${p}-[^\\\\/]*$`).test(f)))
const readPrompt = (b, round) => `${CONTEXT}\n\nYour tiles: ${b.map(t => DIR + '\\' + t + '.png').join(' , ')}\n\nFor EACH tile: Read it, then zoom BOTH model panels before any verdict, because the first view of a tile is scaled down and too small to judge: crop panel 2 ("model, order A") and panel 3 ("model, order B") at full resolution or larger, in pieces if needed, save them under ${SCRATCH}\\<tile>\\${round}\\ as A-<anything>.png and B-<anything>.png, and Read every crop. Zoom further wherever anything looks structured. List every crop you Read in zoom_files. Then report a verdict. "letters" only if you can see at least one shape you would name as a specific Greek letter; "strokes-not-letters" for stroke-like or row-like ink structure you cannot name; "nothing" for blobs, fibres, noise or edge glow. Confidence 0-1. Be honest in both directions: missing real letters and inventing letters from noise are equally bad.`

async function readAll(names, round) {
  const batches = makeBatches(names)
  batches.forEach((b, i) => log(`${round} batch ${i + 1}/${batches.length}: ${b.join(' ')} | control: ${b.filter(t => CONTROLS.has(t)).join(' ') || 'none'}`))
  const reads = await waves(batches, 8, (b) => agent(readPrompt(b, round), { label: `${round}:${b.join(',')}`, phase: 'Read', schema: readSchema(b) }))
  const got = new Map()
  reads.forEach((r) => ((r && r.tiles) || []).forEach(t => got.set(t.tile, { ...t, round })))
  return { batches, got }
}

phase('Read')
const missingCtl = [...CONTROLS].filter(t => !TILES.includes(t))
if (missingCtl.length) log(`controls not in args.tiles, so not read: ${missingCtl.join(' ')}`)
const first = await readAll(TILES, 'read')
// One re-read, fresh agents, for any tile with no verdict or without both model panels zoomed.
const redo = TILES.filter(t => !first.got.has(t) || !zoomedBoth(first.got.get(t)))
let second = { batches: [], got: new Map() }
if (redo.length) {
  log(`re-reading ${redo.length} tiles (no verdict, or panel A or B not zoomed): ${redo.join(' ')}`)
  second = await readAll(redo, 'reread')
}
// A flag never vanishes on a re-read: the stronger verdict of the two rounds stands, the other rides
// along as earlier or later, and a tie goes to the zoomed read.
const RANK = { letters: 2, 'strokes-not-letters': 1, nothing: 0, 'cannot-judge': 0 }
const final = new Map(first.got)
for (const [t, v] of second.got) {
  const u = final.get(t)
  if (!u) { final.set(t, v); continue }
  const keepNew = RANK[v.verdict] > RANK[u.verdict] || (RANK[v.verdict] === RANK[u.verdict] && zoomedBoth(v) && !zoomedBoth(u))
  final.set(t, keepNew ? { ...v, earlier: u } : { ...u, later: v })
}
const all = TILES.filter(t => final.has(t)).map(t => ({ ...final.get(t), zoomed_both: zoomedBoth(final.get(t)) }))
const noVerdict = TILES.filter(t => !final.has(t))
const unzoomed = all.filter(t => !t.zoomed_both).map(t => t.tile)
if (noVerdict.length) log(`NO VERDICT after the re-read: ${noVerdict.join(' ')} (not "nothing": re-run these)`)
if (unzoomed.length) log(`panel A or B still not zoomed after the re-read: ${unzoomed.join(' ')}`)
log(`${all.length} tile verdicts of ${TILES.length}; letters: ${all.filter(t => t.verdict === 'letters').map(t => t.tile).join(' ') || 'none'}; strokes: ${all.filter(t => t.verdict === 'strokes-not-letters').map(t => t.tile).join(' ') || 'none'}`)

const flagged = all.filter(t => t.verdict === 'letters' || t.verdict === 'strokes-not-letters')
const batchesOut = { read: first.batches, reread: second.batches }
if (!VERIFY) {
  log(`args.verify is false: sceptic stage skipped, ${flagged.length} reader flags unverified`)
  return { reads: all, verified: [], batches: batchesOut, no_verdict: noVerdict, unzoomed }
}

phase('Verify')
const LENSES = [
  ['artifact', 'Attack it as an ARTIFACT: edge glow along the black border, papyrus fibre texture, voids or cracks, model blob texture that only looks like a stroke. Check the direct render under the same spot for a crack, void, fold or edge that explains the shape; a blank render explains nothing either way.'],
  ['letterform', `Attack it as LETTERFORM: are the shapes the right size (2-4 mm tall; stroke width as the context says for this model) and do they have the parts a Greek letter needs, consistently, rather than a blob a mind can complete into a letter? Are there rows?`],
  ['consistency', 'Attack it on CONSISTENCY: are the shapes stable in the stronger order (one order is usually the right one and the other often shows little, so a weak or blank other order is no refutation; the two orders showing DIFFERENT shapes in the same place is), and does the structure run with the horizontal fibres the way text rows do? Do NOT count a blank direct render against it: on these scans ink is usually invisible in the render, which is why a model is needed. A render that shows a crack, void or edge where the shape sits does count against it.'],
]
const flaggedJobs = flagged.flatMap(f => LENSES.map(l => ({ f, l })))
const votes = await waves(flaggedJobs, 8, (j) => agent(
  `${CONTEXT}\n\nA reader flagged tile ${DIR}\\${j.f.tile}.png as "${j.f.verdict}" (confidence ${j.f.confidence}), panels ${j.f.which_panels || '?'}, at ${j.f.where || '?'}, guess "${j.f.letters_guess || ''}". Their reasons: ${j.f.reasons}\n\nYour job is to REFUTE that claim. ${j.l[1]} Zoom in as needed. Default to refuted=true if you are unsure. still_letters=true only if, after your attack, you would still name at least one specific Greek letter there; give count_legible.`,
  { label: `verify:${j.f.tile}:${j.l[0]}`, phase: 'Verify', schema: VERDICT }).then(v => ({ tile: j.f.tile, lens: j.l[0], v })))

const byTile = {}
for (const x of votes.filter(Boolean)) (byTile[x.tile] = byTile[x.tile] || []).push(x)
const verified = flagged.map(f => {
  const vs = (byTile[f.tile] || []).filter(x => x.v)
  const survive = vs.filter(x => !x.v.refuted).length
  const lettersKept = vs.filter(x => x.v.still_letters).length
  return { tile: f.tile, reader: f, survive, lettersKept, votes: vs.map(x => ({ lens: x.lens, ...x.v })) }
})
return { reads: all, verified, batches: batchesOut, no_verdict: noVerdict, unzoomed }
