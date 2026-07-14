from rpyfactor.test.conftest import run, run_result_int


def _ints(interp):
    out = []
    n = interp.st().size()
    d = n - 1
    while d >= 0:
        out.append(interp.st().peek_int(d))
        d -= 1
    return out


def test_ifte_cond_quotation_true():
    assert run_result_int("5 [ dup 0 > ] [ 10 ] [ 20 ] ifte") == 10


def test_ifte_cond_quotation_false():
    assert run_result_int("-5 [ dup 0 > ] [ 10 ] [ 20 ] ifte") == 20


def test_ifte_cond_restores_stack_before_then():
    # The condition inspects the top but the then-branch sees the original
    # stack; then reads the value the condition only peeked at.
    assert run_result_int("7 [ dup 3 > ] [ 1 + ] [ 1 - ] ifte") == 8


def test_ifte_cond_deep_stack_preserved():
    # Several parked cells below the condition window must survive restore.
    src = "1 2 3 4 5 [ dup 0 > ] [ ] [ ] ifte + + + +"
    assert run_result_int(src) == 15


def test_ifte_cond_pops_and_repushes_same():
    # Condition pops the top, computes a flag from it, restores it.
    src = "9 8 [ over over > ] [ + ] [ - ] ifte"
    assert run_result_int(src) == 17


def test_ifte_cond_nested():
    src = "3 [ dup 0 > ] [ [ dup 5 < ] [ 100 ] [ 200 ] ifte ] [ 999 ] ifte"
    assert run_result_int(src) == 100


def test_ifte_cond_in_recursion():
    src = """
    : countdown ( n -- )
        [ dup 0 > ] [ 1 - countdown ] [ drop ] ifte ;
    5 countdown 42
    """
    assert run_result_int(src) == 42


def test_ifte_cond_deep_recursion_sum():
    src = """
    : sum ( acc n -- acc )
        [ dup 0 > ] [ dup [ + ] dip 1 - sum ] [ drop ] ifte ;
    0 100 sum
    """
    assert run_result_int(src) == 5050
