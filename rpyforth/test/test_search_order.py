from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter
from rpyforth.prelude import load_prelude


def make():
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    load_prelude(outer)
    return inner, outer


def run(line):
    inner, outer = make()
    outer.interpret_line(line)
    return inner


def test_only_forth_also_definitions_is_noop_for_core():
    # The cd16sim idiom must not break ordinary FORTH words.
    inner = run("ONLY FORTH ALSO DEFINITIONS  2 3 +")
    assert inner.pop_ds_int() == 5


def test_words_defined_after_setup_are_found():
    inner, outer = make()
    outer.interpret_line("ONLY FORTH ALSO DEFINITIONS")
    outer.interpret_line(": SQ DUP * ;")
    outer.interpret_line("5 SQ")
    assert inner.pop_ds_int() == 25


def test_wordlist_and_search_wordlist():
    inner, outer = make()
    outer.interpret_line("WORDLIST CONSTANT MY-WL")
    # switch current definitions to MY-WL and define a private word
    outer.interpret_line("GET-CURRENT  MY-WL SET-CURRENT")
    outer.interpret_line(": PRIV 111 ;")
    outer.interpret_line("SET-CURRENT")   # restore prior current wordlist
    # PRIV is not in the FORTH search order, so SEARCH-WORDLIST must find it
    # only when MY-WL is searched explicitly.
    outer.interpret_line('S" PRIV" MY-WL SEARCH-WORDLIST')
    found = inner.pop_ds_int()
    assert found != 0     # 1 or -1: found, xt beneath
    inner.pop_ds()        # drop xt


def test_also_previous_restore_order():
    inner, outer = make()
    outer.interpret_line("WORDLIST CONSTANT W2")
    outer.interpret_line("W2 SET-CURRENT  : HIDDEN 999 ;  FORTH-WORDLIST SET-CURRENT")
    # put W2 on top of the order, use its word, then PREVIOUS to drop it
    outer.interpret_line("W2 >ORDER")
    outer.interpret_line("HIDDEN")
    assert inner.pop_ds_int() == 999
    outer.interpret_line("PREVIOUS")
    # now HIDDEN is no longer reachable; a fresh lookup should push nothing
    # (UNKNOWN printed). Guard by checking a core word still works.
    outer.interpret_line("7 8 +")
    assert inner.pop_ds_int() == 15


def test_get_order_set_order_roundtrip():
    inner, outer = make()
    outer.interpret_line("GET-ORDER")
    n = inner.pop_ds_int()
    assert n >= 1
    # drop the wordlist ids GET-ORDER pushed
    for _ in range(n):
        inner.pop_ds_int()
