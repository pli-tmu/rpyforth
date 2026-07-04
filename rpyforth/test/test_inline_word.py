from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run_lines(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    for line in lines:
        outer.interpret_line(line)
    return inner, outer


def test_inline_defines_word_like_colon():
    # :inline name body ; behaves structurally like : name body ; (fcp's own
    # profiling fallback is exactly : :inline : ;). Inlining is a perf detail.
    inner, _ = run_lines([
        ":inline piece 7 AND ;",
        ": t 15 piece ;",
        "t",
    ])
    assert inner.pop_ds_int() == (15 & 7)


def test_inline_reported_defined():
    # fcp guards its own :inline definition with [UNDEFINED] :inline [IF] ... .
    # We must report :inline as defined so that block is skipped.
    inner, _ = run_lines([
        "[UNDEFINED] :inline 0= ",
    ])
    # 0= of the [UNDEFINED] flag: defined -> flag 0 -> 0= -> -1 (true)
    assert inner.pop_ds_int() == -1


def test_inline_with_stack_comment_tail():
    # fcp writes :inline otherSide  COLORMASK XOR ; ( color -- ~color )
    inner, _ = run_lines([
        "$30 CONSTANT COLORMASK",
        ":inline otherSide COLORMASK XOR ; ( color -- ~color )",
        ": t $10 otherSide ;",
        "t",
    ])
    assert inner.pop_ds_int() == (0x10 ^ 0x30)
