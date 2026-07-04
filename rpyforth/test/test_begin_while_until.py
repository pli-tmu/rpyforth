from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run_lines(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    for line in lines:
        outer.interpret_line(line)
    return inner, outer


def test_begin_while_until_then_defines():
    # fcp's >goodVar uses BEGIN ... WHILE ... UNTIL THEN: a loop with an early
    # WHILE exit (resolved by THEN) and an UNTIL back-branch to BEGIN.
    inner, outer = run_lines([
        ": walk ( n -- n' )"
        "  BEGIN DUP WHILE"          # while n != 0
        "    DUP 1 = IF EXIT THEN"    # stop at 1
        "    1-"                      # decrement
        "  0 UNTIL"                   # UNTIL false -> loop back to BEGIN
        "  THEN ;",
    ])
    assert "WALK" in outer.dict


def test_begin_while_until_then_runs():
    # Count down from n; the loop decrements until DUP is 0 (WHILE exits) or we
    # short-circuit. Verify it terminates and yields the expected value.
    inner, _ = run_lines([
        ": countdown BEGIN DUP WHILE 1- DUP 0= UNTIL THEN ;",
        "5 countdown",
    ])
    assert inner.pop_ds_int() == 0
