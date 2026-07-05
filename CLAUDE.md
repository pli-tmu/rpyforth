# System prompt (Opus 4.8)

You are working on rpyforth: a Forth interpreter written in RPython, translated
with the PyPy toolchain into a native binary with a meta-tracing JIT. The goal
is to beat gforth-fast on the shootout and appbench suites while staying
ANS-Forth compatible.

## Layout
- `rpyforth/inner_interp.py` — VM: data/return/call stacks, unified byte-addressed
  heap access (buf_get/buf_set, addr>>3), catch frames, Abort/ForthException.
- `rpyforth/outer_interp.py` — parser/compiler: interpret_line, compile-mode words,
  (CF) control-flow replay, DOES> carving with branch-target rebasing, self-inlining.
- `rpyforth/primitives.py` — dictionary primitives; JIT-sensitive hot paths.
- `rpyforth/heap.py` — raw rffi heap (8-byte LE cells, HEAP_SIZE_BYTES = 1<<23).
- `rpyforth/test/` — pytest suite (~450 tests).
- `benchmark/` — run_shootout.py / run_appbench.py harnesses (use `.venv/bin/python`).
- `appbench/appbench-1.4/` — untracked shared benchmark tree. NEVER modify it;
  copy programs to /tmp before instrumenting.

## RPython constraints (code must translate)
- Python 2 subset: no f-strings, no dict/set comprehensions in hot paths, explicit
  types, `%` formatting only.
- `x << LONG_BIT` (64) is UNDEFINED at machine width and poisons JIT range analysis
  ("two integer ranges don't overlap" trace aborts). Use `r_uint`/`intmask` and
  split double-cell math into 32-bit halves (see `_ud_divmod_base`).
- `assert` is compiled out in translated builds — a `None` deref that "works" in
  tests can SIGSEGV translated. Guard hot-path invariants structurally.
- JIT hints: `promote()` only values that are near-constant per trace. Promoting a
  runtime-variable value causes bridge storms (e.g. (LOOP) limits are promoted,
  (+LOOP) limits are not — keep that hybrid).
- Untranslated tests run on top of CPython semantics (unbounded ints); always
  reason about the translated 64-bit behavior separately.

## Commands
- Tests (both modes; check exit codes, never pipe through `| tail`):
  - `PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -q`
  - same with `RPYFORTH_STACK_FRAGMENT=1`
- Build: `make build-jit-stkfrag` (~3 min, produces `./rpyforth-c-stkfrag`).
- Benchmarks: `.venv/bin/python benchmark/run_shootout.py`,
  `.venv/bin/python benchmark/run_appbench.py --iterations 3`.
- gforth reference: `gforth -m 16M ...`; verify semantics of a Forth word against
  gforth before implementing it.

## Workflow
- TDD: write a failing pytest reproducing the behavior (verified against gforth)
  before changing interpreter code.
- After any primitives/inner_interp change, run BOTH test modes; a change can pass
  one and break the other.
- Performance claims need a translated build + benchmark run, not untranslated
  timing.

## Safety rules (absolute — past incidents)
- In a worktree, use `git checkout --detach <hash>` ONLY. NEVER `git update-ref`,
  NEVER `git stash`, NEVER run git against the main checkout — worktrees share
  branch refs and both have moved shared branches before.
- Never modify `appbench/appbench-1.4/` in place; work on /tmp copies.
- `rtk` hook may rewrite git/grep; use `rtk proxy git ...` for raw diffs and
  `/usr/bin/grep` to bypass filtering.
- zsh does not word-split unquoted vars: use `${=args}` for `--jit` params.

## git
- remove AI comments from source code before commits
- keep commit messages one-line without signature
- do not track documents
