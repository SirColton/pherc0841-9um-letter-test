"""runfile.py - a run file is either a result or it says it is not one.

WHY THIS EXISTS. A stale run file was once read as a fresh result: the file read
was the PREVIOUS run's output - complete, plausible, correctly formatted, and
describing a job that had finished before the current one started.

THE MECHANISM, because it matters more than the incident. Scripts here come in two
families:

  REDIRECT FAMILY (most of them).  python x.py > x-run.txt
  The shell truncates x-run.txt at launch, so a premature read shows a SHORT file.
  Wrong, but visibly wrong.

  WRITE-AT-END FAMILY. The script builds its whole report in memory and writes it in
  one call on the last line. The old file therefore survives INTACT for the entire
  run. A read at any point before the last second returns a complete, well-formed,
  stale result with no mark on it at all.

That second family cannot be read safely by being careful, because there is nothing
to be careful about: the stale file and the fresh file are the same shape. So the
fix is not a warning. It is that the old content STOPS EXISTING the moment a job
starts, and that every finished file carries who wrote it and when.

WHAT IT DOES
  claim(path)      overwrites the file NOW with a RUNNING banner - pid, start time,
                   script, its sha1. The stale content is gone before the first
                   voxel is read. If the process dies without finishing, an exit
                   hook rewrites the banner to DIED, so a crash leaves a file that
                   announces the crash instead of impersonating a result.
  done(path, text) atomic replace with a COMPLETE banner plus the report.
  status(path)     what state that file is in, and whether it is older than the
                   script that writes it.

WHAT WOULD MAKE THIS WORTHLESS, decided before writing it:
  - If the banner can be absent from a finished file, the reader has to guess again
    and nothing is gained. So done() is the ONLY way a report gets written, and
    a companion reader tool (not published) refuses to print a run file it cannot
    account for.
  - If claim() ran at the END of a script's setup rather than the start, a stale
    file would still be readable during download and model load - which is most of
    the wall time on the long ones. It is called before any work.
  - A banner is provenance, NOT proof the numbers are right. It says which run
    produced this text. It says nothing about whether that run was any good.

THE ONE THING IT CANNOT DO. A script run directly, with a shell redirect, by
someone who never imports this module, is outside it. That hole is covered from the
other end: the companion reader tool checks the live process table (status() below
does the same), so a report whose script is RUNNING RIGHT NOW is refused whether it
carries a banner or not.
"""
import atexit
import hashlib
import os
import subprocess
import sys
import time

MARK = '###'
FMT = '%Y-%m-%dT%H:%M:%S'
_claims = {}
_fatal = None
_hooked = False


def _sha1(path):
    try:
        with open(path, 'rb') as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:12]
    except OSError:
        return '-'


def _stamp(t):
    return time.strftime(FMT, time.localtime(t))


def _script():
    return os.path.abspath(sys.argv[0] or '?')


def _banner(state, d):
    """The header. Fixed labels, one per line, so a reader can parse it without
    guessing and a person can read it without a tool."""
    L = ['%s RUN  %s  %s' % (MARK, d['script'], state)]
    if state == 'RUNNING':
        L += ['%s pid      %s   started %s   on %s' % (MARK, d['pid'], d['started'], d['host']),
              '%s THIS IS NOT A RESULT. It is the placeholder written when the job' % MARK,
              '%s started, and it replaced whatever was here before so that the old' % MARK,
              '%s run could not be mistaken for this one. Wait, or check the process.' % MARK]
        if d.get('kept'):
            L.append('%s previous the run before this one is kept at %s' % (MARK, d['kept']))
    elif state == 'DIED':
        L += ['%s pid      %s   started %s   on %s' % (MARK, d['pid'], d['started'], d['host']),
              '%s died     %s   after %s' % (MARK, d.get('ended', '?'), d.get('took', '?')),
              '%s THE JOB DID NOT FINISH. %s' % (MARK, d.get('why', 'no reason recorded')),
              '%s There is no result in this file and there is no result on disk.' % MARK]
        if d.get('kept'):
            L.append('%s previous the run before this one is kept at %s' % (MARK, d['kept']))
    else:
        L += ['%s started  %s   finished %s   took %s' % (
                  MARK, d['started'], d.get('ended', '?'), d.get('took', '?')),
              '%s argv     %s' % (MARK, d.get('argv', '')),
              '%s script   sha1 %s   mtime %s' % (MARK, d.get('sha1', '-'), d.get('smtime', '-')),
              '%s host     %s   pid %s' % (MARK, d['host'], d['pid'])]
    L.append(MARK)
    return '\n'.join(L) + '\n'


def _write(path, text):
    """Atomic, so a reader never catches a half-written file. os.replace is atomic
    on Windows too, which the naive open-truncate-write is not."""
    tmp = path + '.writing'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(tmp, path)


def _hook():
    """Remember the exception that killed the run, so the file can name it. By the
    time atexit runs, sys.exc_info() is already cleared - the first version of this
    reported every crash as 'the process ended', which is true and useless."""
    global _hooked
    if _hooked:
        return
    _hooked = True
    prev = sys.excepthook

    def catch(kind, value, tb):
        global _fatal
        _fatal = '%s: %s' % (kind.__name__, value)
        prev(kind, value, tb)

    sys.excepthook = catch


def _keep_previous(path):
    """Move the last report aside before taking the file.

    The run .txt files ARE this project's record - the thing a later reader opens
    to find out what was actually tried. Making a stale read impossible must not be
    paid for by destroying a good report every time a script is launched with a
    typo in its arguments. So the old one is kept beside the new one under a name
    nobody can mistake for the current result, and the banner says where it went.
    """
    if not os.path.exists(path):
        return None
    base, ext = os.path.splitext(path)
    prev = base + '.prev' + ext
    try:
        with open(path, 'rb') as src, open(prev, 'wb') as dst:
            dst.write(src.read())
        return os.path.basename(prev)
    except OSError:
        return None


def claim(path):
    """Take the file NOW. Whatever was in it is gone; that is the point."""
    _hook()
    path = os.path.abspath(path)
    kept = _keep_previous(path)
    s = _script()
    d = {'script': os.path.basename(s), 'pid': os.getpid(), 'host': '-',
         't0': time.time(), 'sha1': _sha1(s),
         'smtime': _stamp(os.path.getmtime(s)) if os.path.exists(s) else '-',
         'argv': ' '.join(os.path.basename(a) if i == 0 else a
                          for i, a in enumerate(sys.argv))}
    d['started'] = _stamp(d['t0'])
    d['kept'] = kept
    _claims[path] = d
    _write(path, _banner('RUNNING', d))
    return path


def done(path, text):
    """Finish. This is the only way a report gets written."""
    path = os.path.abspath(path)
    d = _claims.pop(path, None)
    if d is None:                      # done() without claim() - stamp it anyway
        claim(path)
        d = _claims.pop(path)
    now = time.time()
    d['ended'] = _stamp(now)
    d['took'] = '%.0fs' % (now - d['t0'])
    if not text.endswith('\n'):
        text += '\n'
    _write(path, _banner('COMPLETE', d) + text)
    return path


@atexit.register
def _mark_dead():
    """Anything still claimed at exit did not finish. Say so in the file itself - a
    reader opening it later must not have to reconstruct that from timestamps."""
    for path, d in list(_claims.items()):
        now = time.time()
        d['ended'] = _stamp(now)
        d['took'] = '%.0fs' % (now - d['t0'])
        d['why'] = _fatal or 'the process ended before done() was called - killed, or a clean exit down a path that never writes'
        try:
            _write(path, _banner('DIED', d))
        except OSError:
            pass


# ---- reading side ------------------------------------------------------------

def parse(path):
    """The banner, as a dict, or {} if there is not one."""
    out = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if not line.startswith(MARK):
                    break
                body = line[len(MARK):].strip()
                if not body:
                    continue
                if body.startswith('RUN  '):
                    bits = body[5:].rsplit('  ', 1)
                    out['script'] = bits[0].strip()
                    out['state'] = bits[-1].strip()
                elif ' ' in body:
                    k, v = body.split(' ', 1)
                    out.setdefault(k, v.strip())
    except OSError:
        return {}
    return out if 'state' in out else {}


def _alive(pid):
    """Is that pid still there? None if this machine will not say."""
    try:
        if os.name == 'nt':
            r = subprocess.run(['tasklist', '/FI', 'PID eq %s' % pid, '/NH'],
                               capture_output=True, text=True, timeout=20)
            return str(pid) in (r.stdout or '')
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        return None


def running_scripts():
    """Every .py this machine is RUNNING right now, by basename. The check a
    reader would otherwise make by hand, made once and reused.

    IT MUST BE THE INTERPRETER'S PROCESS, NOT ANY PROCESS THAT MENTIONS THE FILE.
    The first version scanned every command line for tokens ending in .py, and a
    shell launched as `python a.py; python b.py; python c.py` names all three for
    the whole batch - so two finished scripts were reported as running and their
    perfectly good reports were refused. Over-refusal is the safe direction and it
    is still a defect: a check that cries wolf gets ignored, which is how the trap
    this module exists for works in the first place.
    So: only processes whose IMAGE is a python interpreter count, and only the
    FIRST .py token on the line - the script being run, not its arguments.
    """
    rows = []
    try:
        if os.name == 'nt':
            q = ("Get-CimInstance Win32_Process | ForEach-Object "
                 "{ $_.Name + '|' + $_.CommandLine }")
            r = subprocess.run(['powershell', '-NoProfile', '-Command', q],
                               capture_output=True, text=True, timeout=60)
            for line in (r.stdout or '').splitlines():
                name, _, cmd = line.partition('|')
                rows.append((name.strip().lower(), cmd))
        else:
            r = subprocess.run(['ps', '-eo', 'comm=,args='], capture_output=True,
                               text=True, timeout=30)
            for line in (r.stdout or '').splitlines():
                bits = line.split(None, 1)
                if bits:
                    rows.append((os.path.basename(bits[0]).lower(),
                                 bits[1] if len(bits) > 1 else ''))
    except Exception:
        return None
    names = set()
    for image, cmd in rows:
        if not image.startswith('python') and not image.startswith('pythonw'):
            continue
        for tok in cmd.replace('\\', '/').split():
            if tok.endswith('.py'):
                names.add(os.path.basename(tok.strip('"\'')))
                break                      # the script, not its arguments
    return names


def status(path, script=None, live=None):
    """What state is this run file in, and can it be believed as a result?

    Two independent axes, deliberately. WHAT THE FILE SAYS about itself, and
    WHETHER THE WORLD AGREES - a job in flight right now, or a script edited after
    the file was written. The second axis is the one that catches the family this
    module exists for, because those files lie perfectly on the first.
    """
    r = {'path': path, 'exists': os.path.exists(path), 'why': [], 'warn': []}
    if not r['exists']:
        r['state'] = 'MISSING'
        r['ok'] = False
        return r
    r['mtime'] = os.path.getmtime(path)
    r['age'] = time.time() - r['mtime']
    b = parse(path)
    r['banner'] = b
    if not b:
        r['state'] = 'LEGACY'
        r['why'].append('no banner - written before runfile.py existed, so its only '
                        'provenance is its mtime')
    elif b.get('state') == 'RUNNING':
        pid = b.get('pid', '').split()[0] if b.get('pid') else None
        a = _alive(pid) if pid else None
        r['state'] = 'RUNNING' if a else ('DIED' if a is False else 'RUNNING?')
        r['why'].append('the file is a placeholder, not a report')
    else:
        r['state'] = b.get('state', '?')
        if r['state'] == 'DIED':
            r['why'].append('the job that claimed this file did not finish')

    # STALENESS WARNS; IT DOES NOT BLOCK, and the split is deliberate. The failure
    # this module exists for is reading a run file as the result of a job you just
    # launched, and the three signals above catch that completely. "The script was
    # edited afterwards" is a different and weaker thing - most edits here are
    # plumbing, and the .txt files ARE this project's record. Refusing a dozen
    # legitimate records teaches the reader to type --anyway, and a guard everyone
    # bypasses by reflex has stopped guarding. So it rides at the top of the output
    # every single time instead.
    if script and os.path.exists(script):
        smt = os.path.getmtime(script)
        if smt > r['mtime'] + 1:
            r['stale_script'] = True
            r['warn'].append('%s was edited %s, AFTER this file was written, so the '
                             'file describes older code'
                             % (os.path.basename(script), _stamp(smt)))
    if live is None:
        live = running_scripts()
    if live is not None and script and os.path.basename(script) in live:
        r['in_flight'] = True
        r['why'].append('%s IS RUNNING RIGHT NOW - whatever is on disk is the '
                        'previous run' % os.path.basename(script))

    r['ok'] = (r['state'] in ('COMPLETE', 'LEGACY') and not r.get('in_flight'))
    return r
