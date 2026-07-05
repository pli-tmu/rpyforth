from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


def run(line):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(line)
    return inner


def run_and_pop(line):
    return run(line).pop_ds_int()


# --- float/double size words (64-bit gforth: floats=8, dfloats=8, sfloats=4) ---

def test_dfloats():
    assert run_and_pop("1 dfloats") == 8
    assert run_and_pop("3 dfloats") == 24


def test_sfloats():
    assert run_and_pop("1 sfloats") == 4
    assert run_and_pop("3 sfloats") == 12


def test_floats_still_8():
    assert run_and_pop("1 floats") == 8


# --- float-alignment words: faligned/dfaligned align to 8, sfaligned to 4 ---

def test_faligned():
    assert run_and_pop("0 faligned") == 0
    assert run_and_pop("1 faligned") == 8
    assert run_and_pop("8 faligned") == 8
    assert run_and_pop("9 faligned") == 16


def test_dfaligned():
    assert run_and_pop("0 dfaligned") == 0
    assert run_and_pop("1 dfaligned") == 8
    assert run_and_pop("9 dfaligned") == 16


def test_sfaligned():
    assert run_and_pop("0 sfaligned") == 0
    assert run_and_pop("1 sfaligned") == 4
    assert run_and_pop("4 sfaligned") == 4
    assert run_and_pop("5 sfaligned") == 8


# --- alignment words compile inside a colon body ---

def test_faligned_in_definition():
    assert run_and_pop(": fa faligned ; 9 fa") == 16


def test_dfloats_in_definition():
    assert run_and_pop(": df dfloats ; 3 df") == 24


# --- 2CONSTANT works at interpret time and compiled into a defining word ---

def test_2constant_interpret():
    inner = run("7 11 2constant pair pair")
    assert inner.pop_ds_int() == 11
    assert inner.pop_ds_int() == 7


def test_2constant_in_defining_word():
    # end-struct-style: a word that runs 2constant at its own runtime.
    src = ": mk 2constant ; 3 5 mk foo foo"
    inner = run(src)
    assert inner.pop_ds_int() == 5
    assert inner.pop_ds_int() == 3


def test_2variable():
    inner = run("2variable dv 20 30 dv 2! dv 2@")
    assert inner.pop_ds_int() == 30
    assert inner.pop_ds_int() == 20


# --- CS-ROLL: [ 1 ] cs-roll swaps the top two control-flow entries so a THEN
#     after a BEGIN resolves an earlier IF (the gc.fs sweep1 idiom) ---

def test_cs_roll_resolves_earlier_if():
    # Mirrors the gc.fs sweep1 idiom exactly (verified against gforth): an IF
    # with no immediate THEN, a BEGIN loop, and inside it `[ 1 ] [cs-roll] THEN`
    # where THEN resolves the earlier IF. cs-roll runs at compile time via the
    # immediate [cs-roll] wrapper. There is no outer THEN and no fall-through.
    #   n<>0: IF taken, jumps into the loop, pushes 222, EXITs -> 222.
    src = (
        ": [cs-roll] cs-roll ; immediate "
        ": t ( n -- x ) "
        "  0<> if "
        "     begin "
        "        222 "
        "        [ 1 ] [cs-roll] then "
        "        exit "
        "     again ; "
        "5 t"
    )
    assert run_and_pop(src) == 222


def test_cs_roll_false_branch():
    #   n=0: IF not taken; the cs-roll'd THEN target sits just before EXIT, so
    #   control EXITs without pushing 222 (matches gforth: stack stays empty).
    inner = run(
        ": [cs-roll] cs-roll ; immediate "
        ": t ( n -- ) "
        "  0<> if begin 222 [ 1 ] [cs-roll] then exit again ; "
        "42 0 t"
    )
    # only the pre-existing 42 remains; t pushed nothing.
    assert inner.pop_ds_int() == 42
    assert inner.ds_int_size() == 0


# --- POSTPONE of the immediate '(' comment word (compat/assert.fs assertn) ---

def test_postpone_paren_comments_out():
    # foo, when it runs during compilation of bar, executes '(' which eats the
    # following text up to ')'. bar should compute 1 5 + = 6.
    src = (
        ": foo postpone ( ; immediate "
        ": bar 1 foo 2 3 4 ) 5 + ; "
        "bar"
    )
    assert run_and_pop(src) == 6
