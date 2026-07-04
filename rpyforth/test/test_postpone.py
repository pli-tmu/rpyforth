from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter
from rpyforth.prelude import load_prelude


def run(lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    load_prelude(outer)
    for line in lines:
        outer.interpret_line(line)
    return inner


def test_postpone_nonimmediate_primitive_defers_not_executes():
    # An immediate word that POSTPONEs a non-immediate primitive must, when used
    # while compiling another word, compile that primitive into the definition
    # rather than run it immediately (brainless hash! = POSTPONE !).
    inner = run([
        "VARIABLE v",
        ": mystore  POSTPONE ! ; IMMEDIATE",
        ": t  v mystore ;",
        "42 t",
        "v @",
    ])
    assert inner.pop_ds_int() == 42


def test_postpone_nonimmediate_word_defers():
    # Same, for a POSTPONEd non-immediate colon word.
    inner = run([
        ": add2  2 + ;",
        ": plus2  POSTPONE add2 ; IMMEDIATE",
        ": t  10 plus2 ;",
        "t",
    ])
    assert inner.pop_ds_int() == 12


def test_postpone_immediate_word_still_runs_at_enclosing_runtime():
    # POSTPONE of an immediate word emits it directly so it runs when the
    # enclosing word runs (LITERAL is the canonical example).
    inner = run([
        ": lit5  5 POSTPONE LITERAL ; IMMEDIATE",
        ": t  lit5 ;",
        "t",
    ])
    assert inner.pop_ds_int() == 5


def test_state_smart_word_via_postpone():
    # A DOES>/STATE-smart idiom: compile a + when compiling, add when interpreting.
    inner = run([
        ": myplus  STATE @ IF POSTPONE + ELSE + THEN ; IMMEDIATE",
        "3 4 myplus",          # interpret path: adds now
    ])
    assert inner.pop_ds_int() == 7
    inner2 = run([
        ": myplus  STATE @ IF POSTPONE + ELSE + THEN ; IMMEDIATE",
        ": t  10 20 myplus ;",  # compile path: compiles +
        "t",
    ])
    assert inner2.pop_ds_int() == 30
