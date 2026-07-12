from rpyfactor.test.conftest import run, run_result_int
from rpyfactor.values import W_Int, W_List


def test_add():
    assert run_result_int("3 4 +") == 7


def test_dup_swap():
    assert run_result_int("1 2 swap +") == 3


def test_colon_def():
    src = """
    : square ( n -- n ) dup * ;
    5 square
    """
    assert run_result_int(src) == 25


def test_quotation_call():
    assert run_result_int("5 [ 1 + ] call") == 6


def test_dip():
    assert run_result_int("10 20 [ 1 + ] dip +") == 31


def test_if_true():
    src = "t [ 10 ] [ 20 ] if"
    assert run_result_int(src) == 10


def test_if_false():
    src = "f [ 10 ] [ 20 ] if"
    assert run_result_int(src) == 20


def test_if_from_compare():
    assert run_result_int("5 3 > [ 10 ] [ 20 ] if") == 10
    assert run_result_int("2 5 > [ 10 ] [ 20 ] if") == 20


def test_array_reduce():
    src = "{ 1 2 3 4 5 } 0 [ + ] reduce"
    assert run_result_int(src) == 15


def test_times():
    assert run_result_int("0 5 [ 1 + ] times") == 5


def test_cons_uncons():
    interp = run("2 1 { } cons cons uncons")
    assert interp.pop_int_result() == 2
    rest = interp.st().pop()
    assert isinstance(rest, W_List)
    assert len(rest.items) == 1
    assert rest.items[0].val == 1


def test_map():
    src = "{ 1 2 3 } [ dup + ] map"
    interp = run(src)
    lst = interp.st().pop()
    assert [x.val for x in lst.items] == [2, 4, 6]


def test_reduce_sum():
    src = "{ 1 2 3 4 } 0 [ + ] reduce"
    assert run_result_int(src) == 10


def test_filter():
    src = "{ 1 2 3 4 } [ 2 mod 0 = ] filter"
    interp = run(src)
    lst = interp.st().pop()
    assert [x.val for x in lst.items] == [2, 4]


def test_null_zero():
    assert run_result_int("0 empty?") == 1


def test_2dup():
    interp = run("1 2 2dup")
    assert [interp.st().peek_int(d) for d in range(4)] == [2, 1, 2, 1]


def test_2over():
    interp = run("1 2 3 2over")
    assert [interp.st().peek_int(d) for d in range(5)] == [2, 1, 3, 2, 1]


def test_eq_ints():
    assert run_result_int("4 4 =") == 1
    assert run_result_int("4 5 =") == 0


def test_nip():
    assert run_result_int("1 2 nip") == 2


def test_drop():
    assert run_result_int("1 2 drop") == 1


def test_dupd():
    interp = run("7 9 dupd")
    assert [interp.st().peek_int(d) for d in range(3)] == [9, 7, 7]


def test_swapd():
    interp = run("1 2 3 swapd")
    assert [interp.st().peek_int(d) for d in range(3)] == [3, 1, 2]


def test_empty_quotation():
    assert run_result_int("5 [ ] call") == 5
    assert run_result_int("10 20 [ ] dip +") == 30
    assert run_result_int("7 3 [ ] times") == 7


def test_clock():
    interp = run("clock clock")
    t1 = interp.pop_int_result()
    t0 = interp.pop_int_result()
    assert t0 > 0
    assert t1 >= t0


def test_line_comment():
    src = """
    ! this is a comment
    3 4 +  ! trailing
    """
    assert run_result_int(src) == 7


def test_effect_ignored():
    src = """
    : add1 ( n -- n ) 1 + ;
    41 add1
    """
    assert run_result_int(src) == 42


def test_fib_small():
    src = """
    : fib ( n -- n )
        dup 2 < [ drop 1 ] [ dup 2 - fib swap 1 - fib + ] if ;
    10 fib
    """
    assert run_result_int(src) == 89


def test_iota_sum():
    src = """
    : iota ( n -- seq )
        { } swap
        [ dup 0 > ]
        [ dup [ swap cons ] dip 1 - ]
        while drop ;
    5 iota 0 [ + ] reduce
    """
    assert run_result_int(src) == 15
