import time

from rpyfactor.test.conftest import run, run_result_int
from rpyfactor.values import (
    W_Int, W_List, W_Cons, W_Nil, nil_list, w_list_from_items,
    list_length, list_is_empty,
)


def _ints(lst):
    return [x.val for x in lst.items]


def test_nil_singleton():
    assert nil_list() is nil_list()
    assert isinstance(nil_list(), W_Nil)
    assert isinstance(nil_list(), W_List)
    assert list_is_empty(nil_list())
    assert list_length(nil_list()) == 0


def test_from_items_roundtrip():
    lst = w_list_from_items([W_Int(1), W_Int(2), W_Int(3)])
    assert isinstance(lst, W_Cons)
    assert isinstance(lst, W_List)
    assert _ints(lst) == [1, 2, 3]
    assert list_length(lst) == 3
    assert not list_is_empty(lst)


def test_cons_head_tail():
    lst = w_list_from_items([W_Int(2), W_Int(3)])
    consed = W_Cons(W_Int(1), lst)
    assert consed.head.val == 1
    assert consed.tail is lst
    assert _ints(consed) == [1, 2, 3]


def test_cons_prim():
    interp = run("3 2 1 nil cons cons cons")
    lst = interp.st().pop()
    assert isinstance(lst, W_List)
    assert _ints(lst) == [3, 2, 1]


def test_uncons_prim():
    interp = run("2 1 nil cons cons uncons")
    assert interp.pop_int_result() == 2
    rest = interp.st().pop()
    assert isinstance(rest, W_List)
    assert _ints(rest) == [1]


def test_first_rest():
    interp = run("{ 10 20 30 } first")
    assert interp.pop_int_result() == 10
    interp = run("{ 10 20 30 } rest")
    lst = interp.st().pop()
    assert _ints(lst) == [20, 30]


def test_swons():
    interp = run("nil 5 swons")
    lst = interp.st().pop()
    assert _ints(lst) == [5]


def test_concat():
    interp = run("{ 1 2 } { 3 4 5 } concat")
    lst = interp.st().pop()
    assert _ints(lst) == [1, 2, 3, 4, 5]


def test_size():
    assert run_result_int("{ 1 2 3 4 } size") == 4
    assert run_result_int("nil size") == 0


def test_null():
    assert run_result_int("nil null") == 1
    assert run_result_int("{ 1 } null") == 0


def test_small():
    assert run_result_int("nil small") == 1
    assert run_result_int("{ 1 } small") == 1
    assert run_result_int("{ 1 2 } small") == 0


def test_reverse():
    interp = run("{ 1 2 3 } reverse")
    lst = interp.st().pop()
    assert _ints(lst) == [3, 2, 1]


def test_list_literal():
    interp = run("{ 7 8 9 }")
    lst = interp.st().pop()
    assert isinstance(lst, W_List)
    assert _ints(lst) == [7, 8, 9]


def test_empty_literal():
    interp = run("{ }")
    lst = interp.st().pop()
    assert isinstance(lst, W_List)
    assert list_is_empty(lst)


def test_nested_lists():
    interp = run("{ { 1 2 } { 3 4 } }")
    lst = interp.st().pop()
    inner = lst.items
    assert len(inner) == 2
    assert _ints(inner[0]) == [1, 2]
    assert _ints(inner[1]) == [3, 4]


def test_step_over_list():
    src = "0 { 1 2 3 4 } [ + ] step"
    assert run_result_int(src) == 10


def test_map_over_list():
    interp = run("{ 1 2 3 } [ dup + ] map")
    lst = interp.st().pop()
    assert _ints(lst) == [2, 4, 6]


def test_fold_over_list():
    assert run_result_int("{ 1 2 3 4 } 0 [ + ] reduce") == 10


def test_filter_over_list():
    interp = run("{ 1 2 3 4 5 6 } [ 2 mod 0 = ] filter")
    lst = interp.st().pop()
    assert _ints(lst) == [2, 4, 6]


import os
import subprocess
import sys


def _capture_subprocess(source):
    code = (
        "from rpyfactor.interp import Interpreter\n"
        "Interpreter().run_source(%r)\n" % (source,)
    )
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join([root, os.path.join(root, "pypy")])
    out = subprocess.check_output([sys.executable, "-c", code], env=env)
    return out


def test_print_list_format():
    assert _capture_subprocess("{ 1 2 3 } .") == "[1 2 3] "


def test_print_empty_list_format():
    assert _capture_subprocess("nil .") == "[] "


def test_o1_cons_rest_smoke():
    src = """
    : build ( lst n -- lst )
        dup 0 = [ drop ] [ 1 - [ 0 swap cons ] dip build ] if ;
    : strip ( lst n -- lst )
        dup 0 = [ drop ] [ 1 - [ rest ] dip strip ] if ;
    nil 200000 build
    200000 strip
    size
    """
    t0 = time.time()
    assert run_result_int(src) == 0
    assert time.time() - t0 < 20.0
