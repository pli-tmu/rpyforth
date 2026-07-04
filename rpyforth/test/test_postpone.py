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


def test_postpone_if_then_early_exit():
    # brainless tmovegen ?single-move: an IMMEDIATE word POSTPONEs IF/THEN, which
    # are parser tokens rather than dictionary words. When it runs while compiling
    # another word, the IF/THEN control structure must be spliced into that word.
    inner = run([
        ": stop-if  POSTPONE IF  POSTPONE EXIT  POSTPONE THEN ; IMMEDIATE",
        # push 1, then (flag on top) maybe EXIT before dropping it and pushing 2.
        ": t  ( flag -- n ) 1 SWAP stop-if DROP 2 ;",
        "-1 t",   # flag true: EXIT with 1 left on the stack
    ])
    assert inner.pop_ds_int() == 1
    inner2 = run([
        ": stop-if  POSTPONE IF  POSTPONE EXIT  POSTPONE THEN ; IMMEDIATE",
        ": t  ( flag -- n ) 1 SWAP stop-if DROP 2 ;",
        "0 t",    # flag false: fall through, DROP the 1, push 2
    ])
    assert inner2.pop_ds_int() == 2


def test_postpone_if_else_then():
    inner = run([
        ": choose  POSTPONE IF  1 POSTPONE LITERAL  POSTPONE ELSE  2 POSTPONE LITERAL  POSTPONE THEN ; IMMEDIATE",
        ": t  ( flag -- n ) choose ;",
        "-1 t",
    ])
    assert inner.pop_ds_int() == 1
    inner2 = run([
        ": choose  POSTPONE IF  1 POSTPONE LITERAL  POSTPONE ELSE  2 POSTPONE LITERAL  POSTPONE THEN ; IMMEDIATE",
        ": t  ( flag -- n ) choose ;",
        "0 t",
    ])
    assert inner2.pop_ds_int() == 2
