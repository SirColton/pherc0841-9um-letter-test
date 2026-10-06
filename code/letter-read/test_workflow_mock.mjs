// test_workflow_mock.mjs - runs day1-letter-read-workflow.js against MOCK agents (no model is called)
// and checks what a reader is shown and how the run is shaped. Synthetic data only. Prints one JSON
// line {ok, checks:[...], failures:[...]}; exit 1 on any failure. Called by test_letter_read.py.
import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const HERE = dirname(fileURLToPath(import.meta.url))
const WF = process.argv[2] || [join(HERE, '..', 'day1-letter-read-workflow.js'), join(HERE, '..', '..', '..', 'day1-letter-read-workflow.js')].find(p => existsSync(p))
const src = readFileSync(WF, 'utf8')
if (!/^export const meta = \{/.test(src)) throw new Error('the workflow must open with export const meta = {')
const metaText = src.slice(0, src.indexOf('\n}\n') + 3)
if (/\$\{|\.\.\.|\bargs\b|\(/.test(metaText.replace(/'[^']*'/g, "''"))) throw new Error('meta is not a pure literal')
const body = src.replace(/^export const meta = /, 'const meta = ')
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
const run = new AsyncFunction('args', 'agent', 'parallel', 'pipeline', 'log', 'phase', 'budget', 'workflow', body)

const checks = [], failures = []
const ok = (cond, what) => { (cond ? checks : failures).push(what) }

// A synthetic pool: four real tiles on a 2 x 2 grid that all overlap (as w00big's 8 mm grid does), two
// far-away real tiles, and controls whose source windows sit on the same grid.
const PITCH = 9.612
const T = 832
const box = (i, j) => [i, j, i + T, j + T]
const MANIFEST = {
  ok: true, exit_code: 0, problems: [], pitch_um: PITCH,
  tiles: [
    { tile: 't0a11111', control: false, box: box(0, 0), png_px: [T, T] },
    { tile: 't0b22222', control: false, box: box(0, 413), png_px: [T, T] },
    { tile: 't0c33333', control: false, box: box(163, 0), png_px: [T, T] },
    { tile: 't0d44444', control: false, box: box(163, 413), png_px: [T, T] },
    { tile: 't0e55555', control: false, box: box(5000, 5000), png_px: [T, T] },
    { tile: 't0f66666', control: false, box: box(9000, 9000), png_px: [T, T] },
    { tile: 't1a77777', control: true, box: box(0, 0), png_px: [T, T] },
    { tile: 't1b88888', control: true, box: box(0, 413), png_px: [T, T] },
    { tile: 't1c99999', control: true, box: box(163, 0), png_px: [T, T] },
    { tile: 't1daaaaa', control: true, box: box(20000, 20000), png_px: [T, T] },
  ],
}
const CONTROLS = new Set(MANIFEST.tiles.filter(t => t.control).map(t => t.tile))
const BOX = new Map(MANIFEST.tiles.map(t => [t.tile, t.box]))
const ARGS = {
  tiles_dir: 'X:\\pool\\p1\\tiles',
  key_path: 'X:\\pool\\SECRET-key-p1.json',
  reference_path: 'X:\\pool\\SECRET-ref.json',
  out_path: 'X:\\pool\\out\\calls-p1.json',
  per_batch: 3, concurrency: 2, seed: 7,
}

function makeAgent(opt) {
  const state = { calls: [], inflight: 0, maxInflight: 0, prompts: [] }
  const agent = async (prompt, o) => {
    state.calls.push({ label: o.label, phase: o.phase })
    if (o.phase === 'Key') return opt.manifest === undefined ? MANIFEST : opt.manifest
    if (o.phase === 'Write') {
      const text = prompt.split('-----BEGIN-----\n')[1].split('\n-----END-----')[0]
      const d = JSON.parse(text)
      state.written = d
      const box_sum = d.calls.reduce((s, c) => s + c.box.reduce((u, v) => u + v, 0), 0)
      return opt.badWriter ? { written: true, n_calls: d.calls.length + 1, box_sum, n_reads: d.reads.length }
        : { written: true, n_calls: d.calls.length, box_sum, n_reads: d.reads.length }
    }
    // a reader
    state.inflight++
    state.maxInflight = Math.max(state.maxInflight, state.inflight)
    state.prompts.push({ label: o.label, prompt })
    const [round, list] = o.label.split(':')
    const names = list.split(',')
    await new Promise(r => setTimeout(r, 5 + (names[0].charCodeAt(3) % 7)))
    state.inflight--
    const tiles = names.filter(n => !(opt.dropFirst && round === 'read' && n === opt.dropFirst)).map(n => ({
      tile: n,
      letters: CONTROLS.has(n) ? [] : [{ letter: 'alpha', box: [100.4, 200.6, 180, 290], confidence: 0.734 }],
      zoom_files: [`X:\\pool\\p1\\tiles-zoom\\${n}\\${round}\\a-1.png`],
      notes: 'mock',
    }))
    return { tiles }
  }
  return { agent, state }
}
const parallel = thunks => Promise.all(thunks.map(t => t().catch(() => null)))
const pipeline = async () => { throw new Error('not used') }

async function go(args, opt = {}) {
  const { agent, state } = makeAgent(opt)
  const logs = []
  const out = await run(args, agent, parallel, pipeline, m => logs.push(m), () => {}, { total: null, spent: () => 0, remaining: () => Infinity }, null)
  return { out, state, logs }
}

// 1. the normal run
const r1 = await go(ARGS)
const readerPrompts = r1.state.prompts
ok(r1.out.write_ok === true, 'writer check passes when the file holds every call')
ok(r1.out.tiles_read.length === MANIFEST.tiles.length && r1.out.no_read.length === 0, 'every tile read once')
ok(r1.out.calls.length === MANIFEST.tiles.length - CONTROLS.size, 'one call per real tile returned')
ok(r1.out.calls.every(c => typeof c.tile === 'string' && c.letter === 'alpha' && c.box.length === 4 && c.box.every(Number.isInteger)), 'every call carries tile, letter and an integer box')
ok(JSON.stringify(r1.out.calls[0].box) === JSON.stringify([100, 201, 180, 290]), 'boxes are rounded, not moved')
const leaks = readerPrompts.filter(p => ['SECRET', 'key-p1', 'ref.json', 'calls-p1', 'manifest', 'control', 'phase', 'scrambl', 'shuffl', 'null'].some(w => p.prompt.toLowerCase().includes(w.toLowerCase())))
ok(leaks.length === 0, 'no reader prompt names the key, the reference, the output, controls or null kinds' + (leaks.length ? ': ' + leaks[0].label : ''))
const foreign = readerPrompts.filter(p => { const mine = p.label.split(':')[1].split(','); return MANIFEST.tiles.some(t => !mine.includes(t.tile) && p.prompt.includes(t.tile)) })
ok(foreign.length === 0, 'no reader prompt names a tile outside its own batch')
const batches = r1.out.batches.read
ok(batches.every(b => b.filter(n => CONTROLS.has(n)).length <= 1), 'at most one control per batch')
const TOL = 500 / PITCH
const ov = (a, b) => Math.min(a[2], b[2]) - Math.max(a[0], b[0]) > TOL && Math.min(a[3], b[3]) - Math.max(a[1], b[1]) > TOL
ok(batches.every(b => b.every((x, i) => b.every((y, k) => k <= i || !ov(BOX.get(x), BOX.get(y))))), 'no two tiles from overlapping windows share a batch')
ok(batches.every(b => b.length <= ARGS.per_batch) && batches.flat().length === MANIFEST.tiles.length, 'batches respect per_batch and cover every tile once')
ok(r1.state.maxInflight <= ARGS.concurrency && r1.state.maxInflight >= 1, `readers in flight never exceed concurrency (max seen ${r1.state.maxInflight})`)
ok(r1.state.calls.filter(c => c.phase === 'Key').length === 1 && r1.state.calls[0].phase === 'Key', 'the key is read once, before any reader')

// 2. determinism: same args, same batches; another seed, other batches allowed
const r2 = await go(ARGS)
ok(JSON.stringify(r2.out.batches) === JSON.stringify(r1.out.batches), 'same seed gives the same batches')

// 3. concurrency is capped at 8
const r3 = await go({ ...ARGS, concurrency: 20, per_batch: 1 })
ok(r3.state.maxInflight <= 8 && r3.logs.some(l => l.includes('capped at 8')), `concurrency 20 is capped at 8 (max seen ${r3.state.maxInflight})`)
const r3b = await go({ ...ARGS, concurrency: undefined, per_batch: 1 })
ok(r3b.state.maxInflight <= 3 && r3b.state.maxInflight >= 2, `default concurrency is 3 (max seen ${r3b.state.maxInflight})`)

// 4. a failed manifest (reference not locked) stops before any read
const r4 = await go(ARGS, { manifest: { ok: false, exit_code: 5, problems: ['REFERENCE NOT LOCKED: no lock file'], pitch_um: PITCH, tiles: [] } })
ok(r4.out.error && r4.state.prompts.length === 0, 'an unlocked reference stops the run before any reader')

// 5. a tile missing from its first read is re-read by a fresh agent
const r5 = await go(ARGS, { dropFirst: 't0e55555' })
ok(r5.out.batches.reread.flat().includes('t0e55555') && r5.out.tiles_read.includes('t0e55555'), 'a tile with no answer is re-read')

// 6. a writer whose file does not check is retried once and then reported
const r6 = await go(ARGS, { badWriter: true })
ok(r6.out.write_ok === false && r6.state.calls.filter(c => c.phase === 'Write').length === 2, 'a writer mismatch is retried once and reported')

// 7. the written file is what score.py reads
const w = r1.state.written
ok(w && Array.isArray(w.calls) && Array.isArray(w.tiles_read) && w.calls.length === r1.out.calls.length, 'written file has calls and tiles_read')

// 8. paths: a key inside the tiles folder is refused before anything runs
let threw = false
try { await go({ ...ARGS, key_path: 'X:\\pool\\p1\\tiles\\key.json' }) } catch (e) { threw = /inside the tiles folder/.test(String(e)) }
ok(threw, 'a key path inside the tiles folder is refused')

console.log(JSON.stringify({ ok: failures.length === 0, checks, failures }))
process.exit(failures.length ? 1 : 0)
