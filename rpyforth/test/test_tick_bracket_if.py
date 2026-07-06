from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter
from rpyforth.prelude import load_prelude


def run_lines(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    load_prelude(outer)
    for line in lines:
        outer.interpret_line(line)
    return inner, outer


def test_tick_bracket_if_returns_xt():
    # brew's gene engine does `' [IF] ... >gene-evaluated-xt !`; [IF]/[ELSE]/[THEN]
    # must be tickable (real dictionary words), not only lexical dispatch.
    inner, outer = run_lines(["' [IF]"])
    xt = inner.pop_ds_int()
    assert xt >= 0
    assert "[IF]" in outer.dict


def test_tick_bracket_else_then():
    inner, outer = run_lines(["' [ELSE] ' [THEN]"])
    then_xt = inner.pop_ds_int()
    else_xt = inner.pop_ds_int()
    assert then_xt >= 0 and else_xt >= 0


def test_normal_bracket_if_still_conditional():
    # The lexical [IF] path is unaffected: a false flag skips its branch so the
    # word defined in the [ELSE] arm is the one that exists.
    inner, outer = run_lines([
        "0 [IF]",
        ": t 111 ;",
        "[ELSE]",
        ": t 222 ;",
        "[THEN]",
        "t",
    ])
    assert inner.pop_ds_int() == 222
