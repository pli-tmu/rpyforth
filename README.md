# RPyForth

An ANS-Forth system written in [RPython](https://rpython.readthedocs.io/) and
translated with the PyPy toolchain into a native binary with a meta-tracing
JIT compiler.

The interpreter is deliberately written as a straightforward indirect-threaded
VM.  The key runtime structure is a three-tier *metastack*: the top
data-stack cells live in scalar fields (registers inside a trace), the next
cells in a small. JIT-virtualizable frame array, and everything deeper in one
shared spill area The compiler additionally inlines small colon words -- including
ones containing control flow -- at compile time.

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

## Running the tests

The suite runs untranslated on the PyPy2 interpreter. Run it in both stack
configurations:

```sh
make test
RPYFORTH_STACK_FRAGMENT=1 PYTHONPATH=. ./_pypy_binary/bin/python2 ./pypy/pytest.py rpyforth/test -q
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
