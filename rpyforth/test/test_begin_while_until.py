from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run_lines(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    for line in lines:
        outer.interpret_line(line)
    return inner, outer


def test_begin_double_while_repeat_then():
    # ansify.fth xt-skip uses BEGIN .. WHILE .. WHILE .. REPEAT THEN: two forward
    # exits sharing one BEGIN. REPEAT resolves the nearest WHILE and branches back;
    # the trailing THEN resolves the first WHILE.
    # skip-zeros: ( n -- n' ) while n>0 and n even, halve; stop at odd or zero.
    inner, outer = run_lines([
        ": skip-zeros ( n -- n' )"
        "  BEGIN DUP WHILE"            # n != 0
        "    DUP 1 AND 0= WHILE"       # n even
        "    2/"                        # halve
        "  REPEAT THEN ;",
    ])
    assert "SKIP-ZEROS" in outer.dict
    inner, _ = run_lines([
        ": skip-zeros ( n -- n' )"
        "  BEGIN DUP WHILE DUP 1 AND 0= WHILE 2/ REPEAT THEN ;",
        "8 skip-zeros",   # 8->4->2->1 (odd, exits via 2nd WHILE): 1
    ])
    assert inner.pop_ds_int() == 1
    inner, _ = run_lines([
        ": skip-zeros ( n -- n' )"
        "  BEGIN DUP WHILE DUP 1 AND 0= WHILE 2/ REPEAT THEN ;",
        "0 skip-zeros",   # 0 exits via 1st WHILE: 0
    ])
    assert inner.pop_ds_int() == 0


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
