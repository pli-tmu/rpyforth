# RPyForth

An ANS-Forth system written in [RPython](https://rpython.readthedocs.io/) and
translated with the PyPy toolchain into a native binary with a meta-tracing
JIT compiler.

The interpreter is written as an indirect-threaded VM.  The key runtime structure is
a three-area-layout *metastack*: the top data-stack cells live in registers as known as the stack-caching technique,
the next cells in a small JIT-virtualizable frame array, and everything deeper in one
shared spill area.

The repository also contains the benchmark infrastructure used to compare against gforth,
gforth-fast, VFX Forth, and SwiftForth on the shootout and appbench-1.4 suites.

## Requirements

- Linux x86-64
- `python3` (bootstrap and benchmark harnesses)
- `make`, `gcc`, and the usual C toolchain (for RPython translation)

The comparison engines (gforth/gforth-fast, VFX Forth, SwiftForth) are only
needed for benchmarks, not for building or using RPyForth.

## Building

```sh
make build-jit-stkfrag
```

The first build bootstraps everything automatically: it downloads a `pypy2`
binary into `_pypy_binary/`, clones the PyPy source into `pypy/`, and then
translates the interpreter (a few minutes on a typical machine). The result
is a self-contained native binary:

```sh
./rpyforth-c-stkfrag program.fs
```

The translation-time stack representation has one canonical setting:

```sh
RPYFORTH_STACK_LAYOUT=plain
RPYFORTH_STACK_LAYOUT=fragment       # flagship t0/t1 + frame + spill
RPYFORTH_STACK_LAYOUT=frame-only
RPYFORTH_STACK_LAYOUT=ntop4          # also ntop2, ntop8, ntop16
RPYFORTH_STACK_LAYOUT=fragment-float
```

`RPYFORTH_FRAME_SIZE` remains an independent numeric tuning parameter. The
older stack-fragment boolean flags are accepted for compatibility, but new
build and test commands should use `RPYFORTH_STACK_LAYOUT`.

All `RPYFORTH_*` environment variables are parsed once in
`rpyforth/config.py`; interpreter code only sees resolved constants. To inspect
the effective translation settings before a build:

```sh
RPYFORTH_STACK_LAYOUT=fragment PYTHONPATH=. python -m rpyforth.config
```

See [`CONFIGURATION.md`](CONFIGURATION.md) for the complete setting
list, defaults, and precedence.

## Running the tests

The suite runs untranslated on the PyPy2 interpreter. Run it in both stack
configurations:

```sh
make test
RPYFORTH_STACK_LAYOUT=fragment PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -q
```

`make test-factor` runs the rpyfactor suite.

## Benchmarks

Everything runs from the project root through `make`; each target sets up
its own prerequisites on first use (Python 3 venv with matplotlib, the
translated binary if missing, the comparison engines, and the appbench-1.4
sources):

```sh
make bench-shootout        # shootout micro-benchmarks -> compare.pdf
make bench-shootout-curve  # shootout warm-up curves   -> warmup.pdf
make bench-appbench        # appbench cold functional + performance grid -> appbench.pdf
make bench-appbench-curve  # appbench warm steady-state + warm-up curves -> appbench-curve.pdf
```

For finer control (engine subset, iteration count, program subset), call the
harnesses directly:

```sh
.venv/bin/python benchmark/run_shootout.py --help
.venv/bin/python benchmark/run_appbench.py --help
```

Performance measurements should be taken from a translated build, pinned to
a core, with medians over repeated runs; the harnesses handle this.
