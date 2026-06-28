from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run(line):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(line)
    return inner


def test_catch_no_throw():
    inner = run(": ok 5 ;  ' ok CATCH")
    assert inner.pop_ds_int() == 0   # CATCH result code
    assert inner.pop_ds_int() == 5   # ok's value


def test_catch_catches_throw():
    inner = run(": bad 7 THROW ;  ' bad CATCH")
    assert inner.pop_ds_int() == 7


def test_catch_restores_stack():
    # items pushed before THROW are discarded; the pre-CATCH 99 survives
    inner = run("99  : bad 1 2 3 7 THROW ;  ' bad CATCH")
    assert inner.pop_ds_int() == 7    # code
    assert inner.pop_ds_int() == 99   # restored depth


def test_throw_zero_is_noop():
    inner = run(": ok 5 0 THROW ;  ' ok CATCH")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 5


def test_catch_nested():
    inner = run(": bad 9 THROW ;  : mid ['] bad CATCH ;  ' mid CATCH")
    assert inner.pop_ds_int() == 0   # outer: mid returned normally
    assert inner.pop_ds_int() == 9   # inner caught 9
