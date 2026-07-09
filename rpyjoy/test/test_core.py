from rpyjoy.test.conftest import run, run_result_int
from rpyjoy.values import W_Int, W_List, W_Bool


def test_add():
    assert run_result_int("3 4 +") == 7


def test_dup_swap():
    assert run_result_int("1 2 swap +") == 3


def test_define():
    src = """
    DEFINE square == dup * .
    5 square
    """
    assert run_result_int(src) == 25


def test_quotation_i():
    assert run_result_int("5 [1 +] i") == 6


def test_dip():
    assert run_result_int("10 20 [1 +] dip +") == 31


def test_ifte_true():
    src = "5 3 [>] [10] [20] ifte"
    assert run_result_int(src) == 10


def test_ifte_false():
    src = "2 5 [>] [10] [20] ifte"
    assert run_result_int(src) == 20


def test_ifte_flag_form():
    src = "true [10] [20] ifte"
    assert run_result_int(src) == 10
    src = "false [10] [20] ifte"
    assert run_result_int(src) == 20


def test_iota_sum():
    src = """
    DEFINE iota ==
      nil swap
      [ dup 0 > ]
      [ dup [swons] dip pred ]
      while pop .
    5 iota 0 [+] rot fold
    """
    assert run_result_int(src) == 15


def test_times():
    assert run_result_int("0 5 [1 +] times") == 5


def test_cons_uncons():
    interp = run("2 1 nil cons cons uncons")
    assert interp.pop_int_result() == 2
    rest = interp.stack.pop()
    assert isinstance(rest, W_List)
    assert len(rest.items) == 1
    assert rest.items[0].val == 1


def test_map():
    src = "nil 3 swons 2 swons 1 swons [dup dup +] map"
    interp = run(src)
    lst = interp.stack.pop()
    assert [x.val for x in lst.items] == [2, 4, 6]


def test_fold_sum():
    src = "0 [+] nil 4 swons 3 swons 2 swons 1 swons fold"
    assert run_result_int(src) == 10


def test_linrec_factorial():
    src = "5 [null] [succ] [dup pred] [*] linrec"
    assert run_result_int(src) == 120


def test_primrec_factorial():
    src = "6 [1] [*] primrec"
    assert run_result_int(src) == 720


def test_while_countdown():
    src = "5 [dup 0 >] [1 -] while"
    assert run_result_int(src) == 0


def test_filter():
    src = "nil 4 swons 3 swons 2 swons 1 swons [2 rem 0 =] filter"
    interp = run(src)
    lst = interp.stack.pop()
    assert [x.val for x in lst.items] == [2, 4]


def test_null_zero():
    interp = run("0 null")
    v = interp.stack.pop()
    assert isinstance(v, W_Bool)
    assert v.val is True


def test_2dup():
    interp = run("1 2 2dup")
    assert [interp.stack.peek(d).val for d in range(4)] == [2, 1, 2, 1]


def test_2over():
    interp = run("1 2 3 2over")
    assert [interp.stack.peek(d).val for d in range(5)] == [2, 1, 3, 2, 1]


def test_eq_ints():
    interp = run("4 4 =")
    v = interp.stack.pop()
    assert isinstance(v, W_Bool)
    assert v.val is True
    interp = run("4 5 =")
    v = interp.stack.pop()
    assert isinstance(v, W_Bool)
    assert v.val is False


def test_nip():
    assert run_result_int("1 2 nip") == 2


def test_tak_classic():
    src = """
    DEFINE pack3 == nil cons cons cons .
    DEFINE x1yz ==
      dup first pred swap
      rest dup first swap
      rest first
      .
    DEFINE y1zx ==
      dup
      rest first pred
      swap
      dup first
      swap
      rest rest first
      swap
      .
    DEFINE z1xy ==
      dup
      rest rest first pred
      swap
      dup first
      swap
      rest first
      .
    DEFINE tak ==
      [ 2over > ]
      [
        pack3
        dup x1yz tak
        swap
        dup y1zx tak
        swap
        z1xy tak
        tak
      ]
      [ nip nip ]
      ifte .
    8 4 2 tak
    """
    assert run_result_int(src) == 3
