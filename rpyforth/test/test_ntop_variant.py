"""Parametric-NTOP int metastack tests.

The parametric variant (metastack_int_ntop.DSIntMetaStackN) reads EFFECTIVE_NTOP
from RPYFORTH_NTOP at import time, so every check runs in a subprocess with the
flag set (mirroring test_frameonly.py's end-to-end pattern). Each subprocess
drives the class directly for the unit-level layout checks, and a whole Forth
program for the flag-gated call-boundary / CATCH integration.
"""

import os
import subprocess
import sys

import pytest


NTOPS = [2, 4, 8, 16]


def _run_snippet(ntop, body):
    """Run ``body`` in a subprocess with the parametric variant selected at
    EFFECTIVE_NTOP=ntop. ``body`` prints its own results; we return stdout lines."""
    env = dict(os.environ)
    env["RPYFORTH_STACK_FRAGMENT"] = "1"
    env["RPYFORTH_NTOP"] = str(ntop)
    env.pop("RPYFORTH_FRAME_ONLY", None)
    env.pop("RPYFORTH_FLOAT_FRAGMENT", None)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    out = subprocess.check_output(
        [sys.executable, "-c", body], env=env,
        cwd=os.getcwd(), stderr=subprocess.STDOUT,
    )
    return out.decode("utf-8").strip().splitlines()


# ---------------------------------------------------------------------------
# Unit level: drive DSIntMetaStackN directly across scalar/frame/spill tiers.
# ---------------------------------------------------------------------------

_UNIT_PROLOGUE = (
    "from rpyforth.metastack import EFFECTIVE_NTOP, FRAME_SIZE, CALL_WINDOW\n"
    "from rpyforth.metastack_int_ntop import DSIntMetaStackN, NTOP_ACTIVE_MAX\n"
    "N = EFFECTIVE_NTOP\n"
    "AMAX = NTOP_ACTIVE_MAX\n"
)


@pytest.mark.parametrize("ntop", NTOPS)
def test_selected_ntop(ntop):
    lines = _run_snippet(ntop, _UNIT_PROLOGUE + "print(N)\n")
    assert lines[-1] == str(ntop)


@pytest.mark.parametrize("ntop", NTOPS)
def test_push_pop_across_all_tiers(ntop):
    # Straddle the scalar boundary (N), the frame boundary (N+FRAME_SIZE) and
    # spill by pushing well past NTOP_ACTIVE_MAX, then read the whole stack back.
    body = _UNIT_PROLOGUE + (
        "n = 2 * AMAX + 3\n"
        "s = DSIntMetaStackN()\n"
        "for v in range(n): s.push(v)\n"
        "assert s.size() == n, s.size()\n"
        "got = [s.pop() for _ in range(n)]\n"
        "assert got == list(range(n - 1, -1, -1)), got\n"
        "print('OK')\n"
    )
    assert _run_snippet(ntop, body)[-1] == "OK"


@pytest.mark.parametrize("ntop", NTOPS)
def test_peek_poke_straddling_boundaries(ntop):
    # Depths chosen to straddle N-1/N (scalar->frame) and N+FRAME_SIZE
    # (frame->spill): peek every depth, then poke at a scalar, a frame and a
    # spill slot and read them back.
    body = _UNIT_PROLOGUE + (
        "n = AMAX + 5\n"
        "s = DSIntMetaStackN()\n"
        "for v in range(n): s.push(v)\n"
        "for depth in range(n):\n"
        "    assert s.peek(depth) == n - 1 - depth, (depth, s.peek(depth))\n"
        "s.poke(0, 100)\n"
        "s.poke(N - 1, 111)\n"
        "s.poke(N, 222)\n"
        "s.poke(AMAX, 333)\n"
        "assert s.peek(0) == 100\n"
        "assert s.peek(N - 1) == 111\n"
        "assert s.peek(N) == 222\n"
        "assert s.peek(AMAX) == 333\n"
        "print('OK')\n"
    )
    assert _run_snippet(ntop, body)[-1] == "OK"


@pytest.mark.parametrize("ntop", NTOPS)
def test_park_commit_roundtrips(ntop):
    # Park/commit at depths {1, 2, N, N+3, N+12}: fill to each depth, take a call
    # fragment (normalizes to <= CALL_WINDOW cached cells), verify the window
    # tops stay readable and size is preserved, then commit and drain in order.
    body = _UNIT_PROLOGUE + (
        "depths = [1, 2, N, N + 3, N + 12]\n"
        "for depth in depths:\n"
        "    s = DSIntMetaStackN()\n"
        "    for v in range(depth): s.push(v)\n"
        "    s.push_fragment()\n"
        "    assert s.cache_depth <= CALL_WINDOW, (depth, s.cache_depth)\n"
        "    assert s.size() == depth, (depth, s.size())\n"
        "    for k in range(min(depth, CALL_WINDOW)):\n"
        "        assert s.peek(k) == depth - 1 - k, (depth, k, s.peek(k))\n"
        "    s.pop_fragment_commit()\n"
        "    assert s.size() == depth\n"
        "    got = [s.pop() for _ in range(depth)]\n"
        "    assert got == list(range(depth - 1, -1, -1)), (depth, got)\n"
        "print('OK')\n"
    )
    assert _run_snippet(ntop, body)[-1] == "OK"


@pytest.mark.parametrize("ntop", NTOPS)
def test_deep_recursion_chain(ntop):
    body = _UNIT_PROLOGUE + (
        "s = DSIntMetaStackN()\n"
        "depth = N + 12\n"
        "for v in range(depth):\n"
        "    s.push(v)\n"
        "    s.push_fragment()\n"
        "assert s.frag_ptr == depth\n"
        "for _ in range(depth): s.pop_fragment_commit()\n"
        "assert s.frag_ptr == 0\n"
        "got = [s.pop() for _ in range(depth)]\n"
        "assert got == list(range(depth - 1, -1, -1)), got\n"
        "print('OK')\n"
    )
    assert _run_snippet(ntop, body)[-1] == "OK"


@pytest.mark.parametrize("ntop", NTOPS)
def test_snapshot_restore(ntop):
    body = _UNIT_PROLOGUE + (
        "n = AMAX + 2\n"
        "s = DSIntMetaStackN()\n"
        "for v in range(n): s.push(v)\n"
        "snap = s.snapshot()\n"
        "s.push(999); s.push(998)\n"
        "assert s.size() == n + 2\n"
        "s.restore(snap)\n"
        "assert s.size() == n\n"
        "for depth in range(n):\n"
        "    assert s.peek(depth) == n - 1 - depth, (depth, s.peek(depth))\n"
        "print('OK')\n"
    )
    assert _run_snippet(ntop, body)[-1] == "OK"


# ---------------------------------------------------------------------------
# End-to-end: whole Forth programs with the flag-gated dispatch, call-boundary
# parking and CATCH restore paths exercised together.
# ---------------------------------------------------------------------------

def _run_forth(ntop, line):
    body = (
        "from rpyforth.outer_interp import OuterInterpreter\n"
        "from rpyforth.inner_interp import InnerInterpreter\n"
        "inner = InnerInterpreter()\n"
        "outer = OuterInterpreter(inner)\n"
        "outer.interpret_line(%r)\n"
        "vals = []\n"
        "while inner.depth_ds_int() > 0:\n"
        "    vals.append(inner.pop_ds_int())\n"
        "print(' '.join(str(v) for v in vals))\n"
    ) % (line,)
    lines = _run_snippet(ntop, body)
    last = lines[-1] if lines else ""
    if not last:
        return []
    return [int(x) for x in last.split()]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_deep_stack_sum(ntop):
    # Sum more ints than the cache holds, crossing frame->spill.
    n = 40
    prog = " ".join("1" for _ in range(n)) + " " + " ".join("+" for _ in range(n - 1))
    assert _run_forth(ntop, prog) == [n]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_colon_call_boundary(ntop):
    assert _run_forth(ntop, ": add3 + + ; 10 20 30 add3") == [60]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_recursive_fib(ntop):
    prog = (": fib dup 2 < if drop 1 else dup 1 - recurse swap 2 - recurse + then ; 10 fib")
    assert _run_forth(ntop, prog) == [89]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_catch_restores_depth(ntop):
    got = _run_forth(ntop, "99 : bad 1 2 3 7 THROW ; ' bad CATCH")
    assert got == [7, 99]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_catch_no_throw(ntop):
    got = _run_forth(ntop, "5 : good 2 3 + ; ' good CATCH")
    assert got == [0, 5, 5]


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_catch_deep_stack(ntop):
    # THROW from a word that grew the stack past the cache, with a live pre-CATCH
    # tail below it: unwind must restore the scalars + frame + spill pointers.
    pushes = " ".join(str(v) for v in range(20))
    got = _run_forth(ntop, "%s : boom 1 2 3 4 5 42 THROW ; ' boom CATCH" % pushes)
    # top: throw-code 42, then the 20 pre-CATCH cells (19..0 top-first)
    assert got[0] == 42
    assert got[1:] == list(range(19, -1, -1))


@pytest.mark.parametrize("ntop", NTOPS)
def test_e2e_deep_pick(ntop):
    n = 40
    prog = " ".join(str(v) for v in range(n)) + " " + str(n - 1) + " PICK"
    assert _run_forth(ntop, prog)[0] == 0


# ---------------------------------------------------------------------------
# Validation: N=2 parametric must behave identically to the NTOP=2 flagship on
# a mixed program touching scalars, frame, spill, calls and CATCH.
# ---------------------------------------------------------------------------

_MIXED_PROG = (
    ": sq dup * ; "
    ": fib dup 2 < if drop 1 else dup 1 - recurse swap 2 - recurse + then ; "
    ": bad 1 2 3 9 throw ; "
    "1 2 3 4 5 6 7 8 9 10 "
    "5 sq 8 fib ' bad catch "
    "3 pick over rot"
)


def _run_forth_flagship(line):
    env = dict(os.environ)
    env["RPYFORTH_STACK_FRAGMENT"] = "1"
    env.pop("RPYFORTH_NTOP", None)
    env.pop("RPYFORTH_FRAME_ONLY", None)
    env.pop("RPYFORTH_FLOAT_FRAGMENT", None)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p])
    body = (
        "from rpyforth.outer_interp import OuterInterpreter\n"
        "from rpyforth.inner_interp import InnerInterpreter\n"
        "inner = InnerInterpreter()\n"
        "outer = OuterInterpreter(inner)\n"
        "outer.interpret_line(%r)\n"
        "vals = []\n"
        "while inner.depth_ds_int() > 0:\n"
        "    vals.append(inner.pop_ds_int())\n"
        "print(' '.join(str(v) for v in vals))\n"
    ) % (line,)
    out = subprocess.check_output(
        [sys.executable, "-c", body], env=env,
        cwd=os.getcwd(), stderr=subprocess.STDOUT,
    )
    lines = out.decode("utf-8").strip().splitlines()
    last = lines[-1] if lines else ""
    return [int(x) for x in last.split()] if last else []


def test_n2_matches_flagship_mixed():
    flagship = _run_forth_flagship(_MIXED_PROG)
    param = _run_forth(2, _MIXED_PROG)
    assert param == flagship, (param, flagship)
