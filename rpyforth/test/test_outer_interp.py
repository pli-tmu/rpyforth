from rpyforth.objects import W_StringObject, CELL_SIZE_BYTES, W_IntObject, W_FloatObject, W_WordObject, word_from_wid
from rpyforth.outer_interp import OuterInterpreter
from rpyforth.inner_interp import InnerInterpreter


import pytest

def run(line):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(line)
    return inner

def run_and_pop(line):
    return run(line).pop_ds_int()

def run_and_pop_float(line):
    return run(line).pop_ds_float()

def test_basic_primitives():
    assert run_and_pop(": SQUARE DUP * ; 3 SQUARE") == 9
    assert run_and_pop(": INC 1 + ;  5 INC") == 6

def test_ZEROs():
    assert run_and_pop("0 0=") == -1 # True
    assert run_and_pop("5 0=") == 0  # False
    assert run_and_pop("0 0<") == 0  # False
    assert run_and_pop("-128 0<") == -1
    assert run_and_pop("-128 0>") == -0
    assert run_and_pop("47 0>") == -1

def test_STORE_FETCH():
    assert run_and_pop("5 0 !    0 @") == 5
    assert run_and_pop("VARIABLE X    123 X !    X @") == 123
    assert run_and_pop("VARIABLE A    10 A !    A @ 5 + A !    A @") == 15
    assert run_and_pop(""": SQUARE DUP * ;    VARIABLE N 7 N !    N @ SQUARE""") == 49
    assert run_and_pop(""": SQUARE DUP * ;    VARIABLE N
7 N !    N @ SQUARE""") == 49
    assert run_and_pop(""": SQUARE DUP * ;    VARIABLE N
7 N !    N @ SQUARE""") == 49
    assert run_and_pop("VARIABLE N   -42 N !   N @") == -42

def test_cell_primitives():
    cell_bytes = CELL_SIZE_BYTES
    assert run_and_pop("CELL") == cell_bytes
    assert run_and_pop("3 CELLS") == 3 * cell_bytes
    assert run_and_pop("VARIABLE X VARIABLE Y Y X -") == cell_bytes
    assert run_and_pop("VARIABLE X VARIABLE Y X CELL+ Y -") == 0

def test_DROP():
    assert run_and_pop("1 2 DROP") == 1

def test_MAX():
    assert run_and_pop("3 5 MAX") == 5

def test_min():
    assert run_and_pop("3 5 MIN") == 3

def test_abs():
    assert run_and_pop("-3 ABS") == 3
    assert run_and_pop("3 ABS") == 3

def test_negate():
    assert run_and_pop("3 NEGATE") == -3
    assert run_and_pop("-3 NEGATE") == 3

def test_rot():
    assert run_and_pop("1 2 3 ROT") == 1

def test_2dup():
    inner = run("1 2 2DUP")
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1

def test_2drop():
    inner = run_and_pop("1 2 3 2DROP") == 1

def test_2swap():
    inner = run("1 2 3 4 2SWAP")
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 4
    assert inner.pop_ds_int() == 3

def test_2over():
    inner = run("1 2 3 4 2OVER")
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 4
    assert inner.pop_ds_int() == 3
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1

def test_mod():
    assert run_and_pop("10 3 MOD") == 1
    assert run_and_pop("-20 6 MOD") == 4

def test_inc():
    assert run_and_pop("5 1+") == 6
    assert run_and_pop("-1 1+") == 0

def test_dec():
    assert run_and_pop("5 1-") == 4
    assert run_and_pop("0 1-") == -1

def test_BRANCH():
    assert run_and_pop(": Z? 0= IF 1 ELSE 2 THEN ; 0 Z?") == 1
    assert run_and_pop(": Z? 0= IF 1 ELSE 2 THEN ; 7 Z?") == 2
    assert run_and_pop(": T1  1 0= IF 111 ELSE  0 0= IF 222 ELSE 333 THEN THEN ; T1") == 222

def test_EMIT():
    assert run_and_pop('10 65 EMIT') == 10

def test_questiondup():
    inner = run("0 ?DUP")
    assert inner.pop_ds_int() == 0
    inner = run("5 ?DUP")
    assert inner.pop_ds_int() == 5
    assert inner.pop_ds_int() == 5

def test_depth():
    assert run_and_pop("0 1 DEPTH") == 2
    assert run_and_pop("0 DEPTH") == 1
    assert run_and_pop("DEPTH") == 0

def test_rshift():
    assert run_and_pop("1 0 RSHIFT") == 1
    assert run_and_pop("1 1 RSHIFT") == 0
    assert run_and_pop("2 1 RSHIFT") == 1
    assert run_and_pop("4 2 RSHIFT") == 1
    assert run_and_pop("32768 15 RSHIFT") == 1
    #assert run_and_pop("0x8000 0xF RSHIFT") == 1

def test_lshift():
    assert run_and_pop("1 0 LSHIFT") == 1
    assert run_and_pop("1 1 LSHIFT") == 2
    assert run_and_pop("1 2 LSHIFT") == 4
    assert run_and_pop("1 15 LSHIFT") == 32768
    #assert run_and_pop("1 0xF LSHIFT") == 0x8000


def test_lshift_wraps_to_signed_cell():
    # Shifting into the sign bit wraps to a signed cell (not an unbounded long),
    # so results can round-trip through a cell store (brainless hash codes).
    assert run_and_pop("1 63 LSHIFT") == -9223372036854775808
    # Shift count >= cell width yields 0.
    assert run_and_pop("1 64 LSHIFT") == 0
    # A high value that sets the top bits (65535 48 LSHIFT > 2**63) must wrap and
    # still store/re-fetch identically -- this is the brainless hash-code path.
    inner = run("CREATE c 8 ALLOT  65535 48 LSHIFT c !  c @")
    stored = inner.pop_ds_int()
    inner2 = run("65535 48 LSHIFT")
    assert stored == inner2.pop_ds_int()


def test_s_to_d():
    inner = run("1024 S>D")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 1024
    inner = run("-1024 S>D")
    assert inner.pop_ds_int() == -1
    assert inner.pop_ds_int() == -1024

def test_mul_star():
    inner = run("1024 4 M*")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 4096
    inner = run("-1024 4 M*")
    assert inner.pop_ds_int() == -1
    assert inner.pop_ds_int() == -4096
    inner = run("-1024 -4 M*")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 4096
    inner = run("9223372036854775807 2 M*")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == -2

def test_bl():
    assert run_and_pop("BL") == 32

def test_u_mul_star():
    inner = run("1024 4 UM*")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 4096
    inner = run("9223372036854775808 2 UM*")
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 0
    inner = run("18446744073709551615 2 UM*")
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 18446744073709551614
    inner = run("18446744073709551615 18446744073709551615 UM*")
    assert inner.pop_ds_int() == 18446744073709551614
    assert inner.pop_ds_int() == 1

def test_and():
    assert run_and_pop("6 3 AND") == 2
    assert run_and_pop("0 9223372036854775807 AND") == 0
    assert run_and_pop("-1 -1 AND") == -1
    assert run_and_pop("18446744073709551615 18446744073709551615 AND") == 18446744073709551615

def test_or():
    assert run_and_pop("6 3 OR") == 7
    assert run_and_pop("0 9223372036854775807 OR") == 9223372036854775807
    assert run_and_pop("-1 -1 OR") == -1
    assert run_and_pop("18446744073709551615 18446744073709551615 OR") == 18446744073709551615

def test_xor():
    assert run_and_pop("6 3 XOR") == 5
    assert run_and_pop("0 9223372036854775807 XOR") == 9223372036854775807
    assert run_and_pop("-1 -1 XOR") == 0
    assert run_and_pop("18446744073709551615 18446744073709551615 XOR") == 0

def test_2STAR():
    assert run_and_pop("16384 2*") == 32768
    assert run_and_pop("1 2*") == 2
    assert run_and_pop("0 2*") == 0

def test_2SLASH():
    assert run_and_pop("16384 2/") == 8192
    assert run_and_pop("2 2/") == 1
    assert run_and_pop("-1 2/") == -1

def test_SDOUBLE_QUOTE():
    str = "Hello, World!"
    inner = run("S\" Hello, World!\"")
    assert inner.pop_ds_int() == len(str)
    ptr = inner.pop_ds_int()
    # Verify the pointer points to valid buffer entry with the correct string
    buf_entry = inner.buf[ptr]
    assert buf_entry is not None
    assert buf_entry.strval == str

def test_DIV():
    assert run_and_pop("10 3 /") == 3
    assert run_and_pop("-20 6 /") == -4
    assert run_and_pop("-9223372036854775808 -1 /") == 9223372036854775808

def test_mul_slash():
    assert run_and_pop("10 3 2 */") == 15
    assert run_and_pop("-20 6 7 */") == -18

def test_div_mod():
    inner = run("10 3 /MOD")
    assert inner.pop_ds_int() == 3
    assert inner.pop_ds_int() == 1
    inner = run("-20 6 /MOD")
    assert inner.pop_ds_int() == -4
    assert inner.pop_ds_int() == 4

def test_mul_div_mod():
    inner = run("10 2 3 */MOD")
    assert inner.pop_ds_int() == 6
    assert inner.pop_ds_int() == 2
    inner = run("-20 4 6 */MOD")
    assert inner.pop_ds_int() == -14
    assert inner.pop_ds_int() == 4

def test_fm_div_mod():
    inner = run("20 S>D 3 FM/MOD")
    assert inner.pop_ds_int() == 6
    assert inner.pop_ds_int() == 2
    inner = run("-20 S>D 6 FM/MOD")
    assert inner.pop_ds_int() == -4
    assert inner.pop_ds_int() == 4
    inner = run("20 5 M* 3 FM/MOD")
    assert inner.pop_ds_int() == 33
    assert inner.pop_ds_int() == 1
    inner = run("-20 5 M* 6 FM/MOD")
    assert inner.pop_ds_int() == -17
    assert inner.pop_ds_int() == 2

def test_um_div_mod():
    inner = run("0 1 2 UM/MOD")
    assert inner.pop_ds_int() == -9223372036854775808
    assert inner.pop_ds_int() == 0
    inner = run("3 0 2 UM/MOD")
    assert inner.pop_ds_int() == 1
    assert inner.pop_ds_int() == 1
    inner = run("18446744073709551615 2 UM* 2 UM/MOD")
    assert inner.pop_ds_int() == -1
    assert inner.pop_ds_int() == 0  # 36893488147419103230 % 2 = 0

def test_sm_div_rem():
    inner = run("7 S>D 3 SM/REM")
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == 1
    inner = run("-7 S>D 3 SM/REM")
    assert inner.pop_ds_int() == -2
    assert inner.pop_ds_int() == -1
    inner = run("7 S>D -3 SM/REM")
    assert inner.pop_ds_int() == -2
    assert inner.pop_ds_int() == 1
    inner = run("-7 S>D -3 SM/REM")
    assert inner.pop_ds_int() == 2
    assert inner.pop_ds_int() == -1

def test_u_less():
    assert run_and_pop("3 5 U<") == -1  # True
    assert run_and_pop("5 3 U<") == 0   # False
    assert run_and_pop("18446744073709551615 0 U<") == 0  # False
    assert run_and_pop("0 18446744073709551615 U<") == -1  # True
    assert run_and_pop("-1 0 U<") == 0  # False
    assert run_and_pop("0 -1 U<") == -1 # True
    # Negative first operand vs small positive: unsigned so -N is huge, never <.
    # (brainless WITHIN hits this: OVER - >R - R> U< with a negative difference.)
    assert run_and_pop("-46 8 U<") == 0
    assert run_and_pop("8 -46 U<") == -1
    # WITHIN built on U<: 45 is outside [91,99), must be false.
    assert run_and_pop("45 91 99 WITHIN") == 0
    assert run_and_pop("45 21 29 WITHIN") == 0
    assert run_and_pop("95 91 99 WITHIN") == -1

def test_u_dot():
    assert run_and_pop("10 10 U.") == 10

# Floating point tests
def test_float_literals():
    def run_and_pop(prog):
        inner = run(prog)
        return inner.pop_ds_float()
    assert run_and_pop("1.0") == 1.0
    assert run_and_pop("3.14") == 3.14
    assert run_and_pop("-2.5") == -2.5
    assert run_and_pop("0.") == 0.0

def test_float_addition():
    def run_and_pop(prog):
        inner = run(prog)
        return inner.pop_ds_float()

    assert run_and_pop("1.0 2.0 F+") == 3.0
    assert run_and_pop("3.5 2.5 F+") == 6.0
    assert run_and_pop("-1.5 1.5 F+") == 0.0

def test_float_subtraction():
    assert run_and_pop_float("5.0 3.0 F-") == 2.0
    assert run_and_pop_float("10.5 2.5 F-") == 8.0
    assert run_and_pop_float("1.0 5.0 F-") == -4.0

def test_float_multiplication():
    assert run_and_pop_float("3.0 4.0 F*") == 12.0
    assert run_and_pop_float("2.5 2.0 F*") == 5.0
    assert run_and_pop_float("-2.0 3.0 F*") == -6.0

def test_float_division():
    assert run_and_pop_float("10.0 2.0 F/") == 5.0
    assert run_and_pop_float("7.0 2.0 F/") == 3.5
    assert run_and_pop_float("1.0 4.0 F/") == 0.25

def test_float_comparison():
    assert run_and_pop("5.0 3.0 F>") == -1  # True
    assert run_and_pop("2.0 8.0 F>") == 0   # False
    assert run_and_pop("3.0 3.0 F>") == 0   # False

def test_float_swap():
    inner = run("1.0 2.0 FSWAP")
    assert inner.pop_ds_float() == 1.0
    assert inner.pop_ds_float() == 2.0

def test_float_in_colon_def():
    assert run_and_pop_float(": CIRCLE-AREA 3.14159 F* ; 5.0 5.0 F* CIRCLE-AREA") == 78.53975

# DO...LOOP tests
def test_simple_do_loop():
    # : TEST 10 0 DO I LOOP ; should leave 0-9 on stack
    inner = run(": TEST 10 0 DO I LOOP ; TEST")
    results = []
    for i in range(10):
        results.append(inner.pop_ds_int())
    assert results == list(range(9, -1, -1))  # popped in reverse order

    inner = run(": SUM 0 5 0 DO I + LOOP ; SUM")
    assert inner.pop_ds_int() == 10

    run(": TEST 10 0 DO I . LOOP ; TEST")
    assert run_and_pop(": SUM 0 5 0 DO I + LOOP ; SUM") == 10

def test_nested_do_loops():
    inner = run(": NESTED 3 0 DO 3 0 DO I J * LOOP LOOP ; NESTED")
    expected = [0, 0, 0, 0, 1, 2, 0, 2, 4]
    results = []
    for i in range(9):
        results.append(inner.pop_ds_int())
    assert results == expected[::-1]  # reversed because stack

def test_leave_in_loop():
    inner = run(": EARLY 10 0 DO I DUP 5 = IF LEAVE THEN LOOP ; EARLY")
    results = []
    while inner.ds_int_size() > 0:
        results.append(inner.pop_ds_int())
    assert results == [5, 4, 3, 2, 1, 0]

def test_compare_op():
    assert run_and_pop("5 3 >") == -1  # True
    assert run_and_pop("2 8 >") == 0   # False
    assert run_and_pop("3 3 >") == 0   # False

    assert run_and_pop("5 3 <") == 0
    assert run_and_pop("2 8 <") == -1
    assert run_and_pop("3 3 <") == 0

    assert run_and_pop("5 5 =") == -1  # True
    assert run_and_pop("3 7 =") == 0   # False
    assert run_and_pop("0 0 =") == -1  # True

def test_pick():
    assert run_and_pop("10 20 30 0 PICK") == 30
    assert run_and_pop("10 20 30 1 PICK") == 20
    assert run_and_pop("10 20 30 2 PICK") == 10

def test_char_bracket():
    # [CHAR] A should compile character code for 'A'
    assert run_and_pop(": TEST [CHAR] A ; TEST") == ord('A')
    assert run_and_pop(": TEST [CHAR] Z ; TEST") == ord('Z')
    assert run_and_pop(": TEST [CHAR] 0 ; TEST") == ord('0')


def test_float():
    assert run_and_pop_float("1.0") == 1.0
    assert run_and_pop_float("3.14") == 3.14

    assert run_and_pop_float("1.0 2.0 F+") == 3.0
    assert run_and_pop_float("5.0 3.0 F-") == 2.0
    assert run_and_pop_float("3.0 4.0 F*") == 12.0
    assert run_and_pop_float("10.0 2.0 F/") == 5.0

    assert run_and_pop("5.0 3.0 F>") == -1
    assert run_and_pop("2.0 8.0 F>") == 0

def test_CHAR():
    assert run_and_pop("CHAR Hello") == ord('H')
    assert run_and_pop("CHAR ello") == ord('e')

def test_2STORE():
    # Standard 2!: the top cell x2 lands at a-addr, x1 at the next cell.
    inner = run("2VARIABLE buf 10 20 buf 2!")
    assert inner.cell_fetch(0).intval == 20
    assert inner.cell_fetch(CELL_SIZE_BYTES).intval == 10

def test_s2f():
    """Test S>F (integer to float conversion)"""
    assert run_and_pop_float("10 S>F") == 10.0

def test_fstore_ffetch():
    """Test F! and F@"""
    result = run_and_pop_float("FVARIABLE X  3.14E0 X F!  X F@")
    assert abs(result - 3.14) < 0.01

def test_fdup():
    """Test FDUP"""
    inner = run("5.5E0 FDUP")
    f1 = inner.pop_ds_float()
    f2 = inner.pop_ds_float()
    assert f1 == f2 == 5.5

def test_begin_while_repeat():
    """Test BEGIN...WHILE...REPEAT loop"""
    # Count from 0 to 4
    result = run_and_pop(": TEST 0 BEGIN DUP 5 < WHILE 1+ REPEAT ; TEST")
    assert result == 5

def test_to_r():
    """Test >R (to-R) - move value from data stack to return stack"""
    inner = run("42 >R")
    # Data stack should be empty
    assert inner.ds_ptr_ints == 0
    # Return stack should have 42
    result = inner.pop_rs()
    assert result == 42

def test_r_from():
    """Test R> (R-from) - move value from return stack to data stack"""
    inner = run("42 >R R>")
    # Data stack should have 42
    result = inner.pop_ds_int()
    assert result == 42
    # Return stack should be empty
    assert inner.rs_ptr == 0

def test_r_fetch():
    """Test R@ (R-fetch) - copy value from return stack to data stack"""
    inner = run("42 >R R@")
    # Data stack should have 42
    result = inner.pop_ds_int()
    assert result == 42
    # Return stack should still have 42
    result2 = inner.pop_rs()
    assert result2 == 42

def test_2to_r():
    """Test 2>R - move two values from data stack to return stack"""
    inner = run("10 20 2>R")
    # Data stack should be empty
    assert inner.ds_ptr_ints == 0
    # Return stack should have 20 on top, 10 below
    result2 = inner.pop_rs()
    result1 = inner.pop_rs()
    assert result1 == 10
    assert result2 == 20

def test_2r_from():
    """Test 2R> - move two values from return stack to data stack"""
    inner = run("10 20 2>R 2R>")
    # Data stack should have 10 and 20
    result2 = inner.pop_ds_int()
    result1 = inner.pop_ds_int()
    assert result1 == 10
    assert result2 == 20
    # Return stack should be empty
    assert inner.rs_ptr == 0

def test_2r_fetch():
    """Test 2R@ - copy two values from return stack to data stack"""
    inner = run("10 20 2>R 2R@")
    # Data stack should have 10 and 20
    result2 = inner.pop_ds_int()
    result1 = inner.pop_ds_int()
    assert result1 == 10
    assert result2 == 20
    # Return stack should still have 10 and 20
    result2_rs = inner.pop_rs()
    result1_rs = inner.pop_rs()
    assert result1_rs == 10
    assert result2_rs == 20

def test_return_stack_complex():
    """Test complex combination of return stack operations"""
    # Test: 1 2 3 >R >R >R R> R> R>
    # Should result in: 3 2 1
    inner = run("1 2 3 >R >R >R R> R> R>")
    result1 = inner.pop_ds_int()
    result2 = inner.pop_ds_int()
    result3 = inner.pop_ds_int()
    assert result1 == 3
    assert result2 == 2
    assert result3 == 1

# Data Space Tests

def test_here():
    """Test HERE - return current data space pointer"""
    inner = run("HERE")
    addr1 = inner.pop_ds_int()
    # HERE should return a valid address
    assert addr1 >= 0

def test_comma():
    """Test , (comma) - store value at HERE and increment"""
    inner = run("HERE  42 ,  HERE")
    addr2 = inner.pop_ds_int()
    addr1 = inner.pop_ds_int()
    # addr2 should be addr1 + cell_size_bytes
    assert addr2 == addr1 + inner.cell_size_bytes
    # Check that 42 was stored at addr1
    stored_val = inner.cell_fetch(addr1)
    assert stored_val.intval == 42

def test_c_comma():
    """Test C, - store character at HERE and increment by 1"""
    inner = run("HERE  65 C,  HERE")
    addr2 = inner.pop_ds_int()
    addr1 = inner.pop_ds_int()
    # addr2 should be addr1 + 1
    assert addr2 == addr1 + 1

def test_allot():
    """Test ALLOT - allocate n address units"""
    inner = run("HERE  10 ALLOT  HERE")
    addr2 = inner.pop_ds_int()
    addr1 = inner.pop_ds_int()
    # addr2 should be addr1 + 10
    assert addr2 == addr1 + 10

# Dictionary Tests

def test_create():
    """Test CREATE - create a new word with data field"""
    result = run_and_pop("CREATE MYDATA  MYDATA")
    # MYDATA should push its address
    assert result >= 0

def test_create_with_comma():
    """Test CREATE with , to store data"""
    result = run_and_pop("CREATE MYVAR  123 ,  456 ,  MYVAR")
    # MYVAR should push its address
    addr = result
    # We need to access the inner interpreter to fetch values
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("CREATE MYVAR2  123 ,  456 ,  MYVAR2")
    addr = inner.pop_ds_int()
    # Fetch the first value
    val1 = inner.cell_fetch(addr)
    assert val1.intval == 123
    # Fetch the second value
    addr2 = addr + CELL_SIZE_BYTES
    val2 = inner.cell_fetch(addr2)
    assert val2.intval == 456

def test_find():
    """Test FIND - find a word in the dictionary (standard: c-addr -- xt flag)."""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" DUP" DROP')
    outer.interpret_line("FIND")
    flag = inner.pop_ds_int()
    xt = inner.pop_ds_int()
    assert word_from_wid(xt).name == "DUP"
    assert flag == -1

def test_find_not_found():
    """Test FIND with non-existent word (standard: leaves c-addr 0)."""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" NOTAWORD" DROP')
    outer.interpret_line("FIND")
    flag = inner.pop_ds_int()
    caddr = inner.pop_ds_int()
    assert flag == 0

def test_execute():
    """Test EXECUTE - execute an execution token"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" DUP" DROP')
    outer.interpret_line("FIND")
    flag = inner.pop_ds_int()
    xt = inner.pop_ds_int()
    inner.push_ds_int(42)
    inner.push_ds_int(xt)
    outer.interpret_line("EXECUTE")
    val2 = inner.pop_ds_int()
    val1 = inner.pop_ds_int()
    assert val1 == 42
    assert val2 == 42

def test_to_body():
    """Test >BODY - get body address from execution token"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("VARIABLE MYVAR")
    outer.interpret_line('S" MYVAR" DROP')
    outer.interpret_line("FIND")
    flag = inner.pop_ds_int()
    xt = inner.pop_ds_int()
    inner.push_ds_int(xt)
    outer.interpret_line(">BODY")
    body_addr = inner.pop_ds_int()
    outer.interpret_line("MYVAR")
    var_addr = inner.pop_ds_int()
    assert body_addr == var_addr

# Source Input Tests

def test_source():
    """Test SOURCE - returns current input buffer"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    test_line = "1 2 3 SOURCE"
    outer.interpret_line(test_line)
    # SOURCE pushes ( c-addr u )
    u = inner.pop_ds_int()
    caddr = inner.pop_ds_int()
    # Should return the length of the line
    assert u == len(test_line)

def test_to_in():
    """>IN returns the address of the runtime parse cursor (a token index).
    After the interpreter has consumed the >IN token itself, the cursor is 1."""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(">IN")
    addr = inner.pop_ds_int()
    pos = inner.cell_fetch(addr)
    assert isinstance(pos, W_IntObject)
    assert pos.intval == 1

# Special Tests

def test_tick():
    """Test ' (tick) - get execution token of next word"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("' DUP")
    xt = inner.pop_ds_int()
    assert word_from_wid(xt).name == "DUP"

def test_paren_comment():
    """Test ( (paren) - comment word"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # The parenthetical comment should be ignored
    outer.interpret_line("1 ( this is a comment ) 2")
    val2 = inner.pop_ds_int()
    val1 = inner.pop_ds_int()
    assert val1 == 1
    assert val2 == 2

def test_tick_execute():
    """Test ' (tick) with EXECUTE"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("' DUP")
    xt = inner.pop_ds_int()
    inner.push_ds_int(42)
    inner.push_ds_int(xt)
    outer.interpret_line("EXECUTE")
    val2 = inner.pop_ds_int()
    val1 = inner.pop_ds_int()
    assert val1 == 42
    assert val2 == 42

# Memory Access Tests

def test_plusstore():
    """Test +! - add to memory location"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("VARIABLE X  10 X !  5 X +!  X @")
    result = inner.pop_ds_int()
    assert result == 15

def test_2fetch():
    """Test 2@ - fetch cell pair"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("2VARIABLE BUF  10 20 BUF 2!  BUF 2@")
    x2 = inner.pop_ds_int()
    x1 = inner.pop_ds_int()
    assert x1 == 10
    assert x2 == 20

def test_c_store_fetch():
    """Test C! and C@ - character store and fetch"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("VARIABLE CBUF  65 CBUF C!  CBUF C@")
    result = inner.pop_ds_int()
    assert result == 65

def test_char_plus():
    """Test CHAR+ - increment address by character size"""
    result = run_and_pop("10 CHAR+")
    assert result == 11

def test_chars():
    """Test CHARS - convert character count to address units"""
    result = run_and_pop("5 CHARS")
    assert result == 5

def test_align():
    """Test ALIGN - align data space pointer"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Set HERE to unaligned position
    outer.interpret_line("HERE  1 C,  ALIGN  HERE")
    here_after = inner.pop_ds_int()
    here_before = inner.pop_ds_int()
    # After ALIGN, HERE should be aligned to cell boundary
    assert here_after % CELL_SIZE_BYTES == 0

def test_aligned():
    """Test ALIGNED - return aligned address"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("1 ALIGNED")
    result = inner.pop_ds_int()
    # Should be aligned to cell boundary
    assert result % CELL_SIZE_BYTES == 0

# Parsing Tests

def test_count():
    """Test COUNT - convert counted string to addr/len"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Create a counted string manually: length byte then the characters, all
    # written with C, (COUNT reads the length as a char/byte, per gforth).
    outer.interpret_line("HERE  3 C,")
    addr = inner.pop_ds_int()
    outer.interpret_line("65 C,  66 C,  67 C,")
    inner.push_ds_int(addr)
    outer.interpret_line("COUNT")
    u = inner.pop_ds_int()
    caddr2 = inner.pop_ds_int()
    # Length should be 3
    assert u == 3
    # caddr2 should be addr + 1 char (skipping the length byte)
    assert caddr2 == addr + 1

def test_word():
    """Test WORD - parse word delimited by character"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Parse a word delimited by space (32)
    # Use separate calls to avoid "World" being executed
    outer.interpret_line("32 WORD Hello")
    caddr = inner.pop_ds_int()
    # caddr points to a counted string in character (byte) space
    length = inner.char_fetch(caddr)
    assert length == 5  # "Hello"

def test_word_count():
    """Test WORD with COUNT"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Parse a word and use COUNT to get addr/len
    outer.interpret_line("32 WORD Test COUNT")
    u = inner.pop_ds_int()
    caddr2 = inner.pop_ds_int()
    # Length should be 4 ("Test")
    assert u == 4

# Pictured Numeric Output Tests

def test_PNO():
    # #S expects double-cell number (ud.lo ud.hi), so push 0 as high-order cell
    def run_and_pop(line):
        inner = InnerInterpreter()
        outer = OuterInterpreter(inner)
        outer.interpret_line(line)
        return inner.pop_ds()
    assert run_and_pop("DECIMAL  12345 0 <# #S #>").strval == '12345'
    # In HEX base the literal 255 is hex 0x255, printed back as "255" (matches
    # gforth). Use a decimal magnitude to exercise hex digit output.
    assert run_and_pop("HEX      0FF 0   <# #S #>").strval == 'FF'
    # 5 is not a binary digit; take the magnitude in decimal, then format binary.
    assert run_and_pop("DECIMAL 5 0  BINARY  <# #S #>").strval == '101'


def test_sign_negative():
    """Test SIGN - adds minus sign for negative numbers"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Use SIGN with a negative number
    # #S expects (ud.lo ud.hi) on stack, so 123 0 gives ud=123
    outer.interpret_line("<# -1 SIGN 123 0 #S #> TYPE")
    # This should output "-123" (the SIGN adds -, then #S converts 123)
    # We can't easily test TYPE output, but we can verify SIGN doesn't crash

def test_sign_positive():
    """Test SIGN - no sign for positive numbers"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Use SIGN with a positive number
    outer.interpret_line("<# 1 SIGN 123 0 #S #>")
    result = inner.pop_ds()
    result.strval == '123'

def test_sign_in_pno():
    """Test SIGN within pictured numeric output"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Complete PNO example with SIGN
    outer.interpret_line("<# -5 SIGN 0 0 #>")
    w_str = inner.pop_ds()
    assert isinstance(w_str, W_StringObject)

# System Tests

def test_fill():
    """Test FILL - fill memory with character"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Fill 5 bytes starting at HERE with 'A' (65)
    outer.interpret_line("HERE 5 65 FILL")

def test_move():
    """Test MOVE - copy memory region"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Store values using VARIABLE to get proper addresses
    outer.interpret_line("VARIABLE SRC1  VARIABLE SRC2  VARIABLE SRC3")
    outer.interpret_line("10 SRC1 !  20 SRC2 !  30 SRC3 !")
    # Get source address and create destination
    outer.interpret_line("VARIABLE DST")
    # Move 3 bytes from SRC1 to DST
    outer.interpret_line("SRC1 DST 3 MOVE")
    # Verify first value was copied
    outer.interpret_line("DST @")
    result = inner.pop_ds_int()
    assert result == 10

def test_state():
    """Test STATE - get interpreter state"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # In interpret mode, STATE should return address with 0
    outer.interpret_line("STATE @")
    state_val = inner.pop_ds_int()
    assert state_val == 0

def test_evaluate():
    """Test EVALUATE - evaluate string as Forth"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Create a string and evaluate it
    outer.interpret_line('S" 1 2 +"')
    outer.interpret_line("EVALUATE")
    result = inner.pop_ds_int()
    assert result == 3

def test_abort_quote_false():
    """Test ABORT\" with false condition - should not abort"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Push some values
    outer.interpret_line("1 2 3")
    # False flag should not abort
    outer.interpret_line('0 ABORT" This should not print"')
    # Stack should still have values
    assert inner.ds_int_size() == 3

def test_abort_quote_true():
    """Test ABORT\" with true condition - should abort"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Push some values
    outer.interpret_line("1 2 3")
    # True flag should abort and clear stack
    outer.interpret_line('-1 ABORT" Error occurred"')
    # Stack should be cleared
    assert inner.ds_int_size() == 0


# ============================================
# Tests for newly implemented Forth 2012 Core words
# ============================================

# Division Tests

def test_divmod():
    """Test /MOD - division with remainder"""
    inner = run("10 3 /MOD")
    quot = inner.pop_ds_int()
    rem = inner.pop_ds_int()
    assert quot == 3
    assert rem == 1

def test_starslash():
    """Test */ - multiply then divide"""
    assert run_and_pop("10 3 2 */") == 15  # (10*3)/2

def test_starslashmod():
    """Test */MOD - multiply then divide with remainder"""
    inner = run("10 3 4 */MOD")
    quot = inner.pop_ds_int()
    rem = inner.pop_ds_int()
    # (10*3)/4 = 30/4 = 7 remainder 2
    assert quot == 7
    assert rem == 2

def test_fmslashmod():
    """Test FM/MOD - floored division"""
    inner = run("7 0 3 FM/MOD")
    quot = inner.pop_ds_int()
    rem = inner.pop_ds_int()
    assert quot == 2
    assert rem == 1

def test_smslashrem():
    """Test SM/REM - symmetric division"""
    inner = run("7 0 3 SM/REM")
    quot = inner.pop_ds_int()
    rem = inner.pop_ds_int()
    assert quot == 2
    assert rem == 1

def test_umslashmod():
    """Test UM/MOD - unsigned division"""
    inner = run("10 0 3 UM/MOD")
    quot = inner.pop_ds_int()
    rem = inner.pop_ds_int()
    assert quot == 3
    assert rem == 1

# Bitwise Tests

def test_invert():
    """Test INVERT - bitwise NOT"""
    assert run_and_pop("0 INVERT") == -1
    assert run_and_pop("-1 INVERT") == 0

# Comparison Tests

def test_u_less():
    """Test U< - unsigned less than"""
    assert run_and_pop("1 2 U<") == -1  # True
    assert run_and_pop("2 1 U<") == 0   # False
    assert run_and_pop("-1 1 U<") == 0  # -1 is large when unsigned

# Control Flow Tests

def test_plusloop():
    """Test +LOOP with positive increment"""
    result = run_and_pop(": TEST 0 10 0 DO I + 2 +LOOP ; TEST")
    assert result == 20  # 0+2+4+6+8 = 20

def test_again():
    """Test BEGIN...AGAIN loop (needs EXIT to break)"""
    result = run_and_pop(": TEST 0 BEGIN 1+ DUP 5 = IF EXIT THEN AGAIN ; TEST")
    assert result == 5

def test_until():
    """Test BEGIN...UNTIL loop"""
    result = run_and_pop(": TEST 0 BEGIN 1+ DUP 5 = UNTIL ; TEST")
    assert result == 5

def test_unloop():
    """Test UNLOOP - remove loop parameters from return stack"""
    # Keep I on stack before comparison by duplicating it
    result = run_and_pop(": TEST 5 0 DO I DUP 3 = IF UNLOOP EXIT THEN DROP LOOP 99 ; TEST")
    assert result == 3  # Should exit when I=3

# Compilation Tests

def test_immediate():
    """Test IMMEDIATE - mark word as immediate"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Define a word and mark it immediate
    outer.interpret_line(": TWICE 2 * ;")
    outer.interpret_line("IMMEDIATE")
    # The last defined word should be marked immediate
    assert outer.last_word.immediate == True

def test_literal():
    """Test LITERAL - compile literal at compile time"""
    result = run_and_pop(": TEST [ 5 3 + ] LITERAL ; TEST")
    assert result == 8

def test_bracket():
    """Test [ and ] - switch between interpret and compile modes"""
    result = run_and_pop(": TEST [ 2 3 + ] LITERAL 10 + ; TEST")
    assert result == 15  # (2+3) + 10

def test_bracket_tick():
    """Test ['] - compile execution token"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(": TEST ['] DUP ; TEST")
    xt = inner.pop_ds_int()
    assert word_from_wid(xt).name == "DUP"

# Number Conversion Tests

def test_base():
    """Test BASE - returns address of base variable"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("BASE@")
    result = inner.pop_ds_int()
    # Default base should be 10 (decimal)
    assert result == 10

def test_base_change():
    """Test changing BASE via HEX/DECIMAL"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line("HEX")  # Set to hex
    outer.interpret_line("BASE@")
    result = inner.pop_ds_int()
    assert result == 16
    outer.interpret_line("DECIMAL")  # Set back to decimal
    outer.interpret_line("BASE@")
    result2 = inner.pop_ds_int()
    assert result2 == 10

# I/O Tests

def test_space():
    """Test SPACE - output single space"""
    # Just verify it doesn't crash
    run("SPACE")

def test_spaces():
    """Test SPACES - output multiple spaces"""
    # Just verify it doesn't crash
    run("5 SPACES")

# Environment Tests

def test_environment_core():
    """Test ENVIRONMENT? with CORE query"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" CORE" ENVIRONMENT?')
    flag = inner.pop_ds_int()
    result = inner.pop_ds_int()
    assert flag == -1  # True
    assert result == -1  # CORE is present

def test_environment_stack_cells():
    """Test ENVIRONMENT? with STACK-CELLS query"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" STACK-CELLS" ENVIRONMENT?')
    flag = inner.pop_ds_int()
    result = inner.pop_ds_int()
    assert flag == -1  # True
    assert result == 64  # Stack size

def test_environment_unknown():
    """Test ENVIRONMENT? with unknown query"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line('S" UNKNOWN-QUERY" ENVIRONMENT?')
    flag = inner.pop_ds_int()
    assert flag == 0  # False - unknown


# RECURSE Tests

def test_recurse_factorial():
    """Test RECURSE for recursive factorial"""
    result = run_and_pop(": FACT DUP 1 > IF DUP 1- RECURSE * THEN ; 5 FACT")
    assert result == 120  # 5! = 120

def test_recurse_countdown():
    """Test RECURSE for countdown"""
    result = run_and_pop(": COUNTDOWN DUP 0 > IF 1- RECURSE THEN ; 5 COUNTDOWN")
    assert result == 0

# Compiled ABORT" Tests

def test_abort_quote_compiled_false():
    """Test compiled ABORT\" with false condition"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(': TEST 0 ABORT" Should not abort" 42 ;')
    outer.interpret_line("TEST")
    result = inner.pop_ds_int()
    assert result == 42

def test_abort_quote_compiled_true():
    """Test compiled ABORT\" with true condition"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    outer.interpret_line(': TEST -1 ABORT" Aborted!" 42 ;')
    # Push a marker to verify stack gets cleared
    outer.interpret_line("99")
    outer.interpret_line("TEST")
    # After abort, stack should be cleared
    assert inner.ds_ptr_ints == 0


def test_utime():
    """Test UTIME - returns current time in microseconds as double-cell"""
    inner = run("UTIME")
    # UTIME pushes ( d.low d.high )
    high = inner.pop_ds_int()
    low = inner.pop_ds_int()
    # Reconstruct the double-cell value
    # On 64-bit systems, high should typically be 0 for reasonable times
    # The value should be positive and represent a reasonable timestamp
    assert high >= 0
    assert low >= 0
    # Combined value should be non-zero (current time)
    usecs = low + (high << 64)
    assert usecs > 0


def test_utime_increases():
    """Test that UTIME increases over time"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Define a word that does some work
    outer.interpret_line(": DELAY 1000 0 DO LOOP ;")
    # Get first timestamp
    outer.interpret_line("UTIME")
    high1 = inner.pop_ds_int()
    low1 = inner.pop_ds_int()
    # Do some work
    outer.interpret_line("DELAY")
    # Get second timestamp
    outer.interpret_line("UTIME")
    high2 = inner.pop_ds_int()
    low2 = inner.pop_ds_int()
    # Second should be >= first (time should not go backwards)
    time1 = low1 + (high1 << 64)
    time2 = low2 + (high2 << 64)
    assert time2 >= time1


def test_cputime():
    """Test CPUTIME - returns CPU times as two double-cells"""
    inner = run("CPUTIME")
    # CPUTIME pushes ( duser.low duser.high dsystem.low dsystem.high )
    sys_high = inner.pop_ds_int()
    sys_low = inner.pop_ds_int()
    user_high = inner.pop_ds_int()
    user_low = inner.pop_ds_int()
    # Values should be non-negative
    assert user_high >= 0
    assert user_low >= 0
    assert sys_high >= 0
    assert sys_low >= 0


def test_cputime_user_increases():
    """Test that CPUTIME user time increases with work"""
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    # Define a word that does CPU-intensive work
    outer.interpret_line(": WORK 10000 0 DO I DROP LOOP ;")
    # Get first CPU time
    outer.interpret_line("CPUTIME")
    sys_high1 = inner.pop_ds_int()
    sys_low1 = inner.pop_ds_int()
    user_high1 = inner.pop_ds_int()
    user_low1 = inner.pop_ds_int()
    # Do some CPU-intensive work
    outer.interpret_line("WORK")
    # Get second CPU time
    outer.interpret_line("CPUTIME")
    sys_high2 = inner.pop_ds_int()
    sys_low2 = inner.pop_ds_int()
    user_high2 = inner.pop_ds_int()
    user_low2 = inner.pop_ds_int()
    # User time should increase (or at least not decrease)
    user1 = user_low1 + (user_high1 << 64)
    user2 = user_low2 + (user_high2 << 64)
    assert user2 >= user1


def test_d_plus():
    """Test D+ - add two double-cell numbers"""
    # 100 as double (100 0) + 200 as double (200 0) = 300 as double (300 0)
    inner = run("100 S>D 200 S>D D+")
    high = inner.pop_ds_int()
    low = inner.pop_ds_int()
    assert low == 300
    assert high == 0


def test_d_minus():
    """Test D- - subtract two double-cell numbers"""
    # 500 as double (500 0) - 200 as double (200 0) = 300 as double (300 0)
    inner = run("500 S>D 200 S>D D-")
    high = inner.pop_ds_int()
    low = inner.pop_ds_int()
    assert low == 300
    assert high == 0


def test_d_minus_with_utime():
    """Test D- with UTIME for elapsed time measurement"""
    # Use run() helper which initializes primitives including DO/LOOP
    # Test the actual D- usage pattern for timing
    inner = run("UTIME 100 S>D D+ UTIME 2SWAP D-")
    # Result should be elapsed microseconds (very small positive or zero)
    high = inner.pop_ds_int()
    low = inner.pop_ds_int()
    elapsed = low + (high << 64)
    # Elapsed time should be non-negative
    assert elapsed >= 0


def test_argc():
    """Test ARGC primitive returns number of command-line arguments."""
    inner = InnerInterpreter()
    inner.argv = ["10", "20"]
    outer = OuterInterpreter(inner)
    outer.interpret_line("ARGC")
    assert inner.pop_ds_int() == 2


def test_argv():
    """Test ARGV primitive returns argument as a counted string."""
    inner = InnerInterpreter()
    inner.argv = ["10"]
    outer = OuterInterpreter(inner)
    # Parse the first argument as a number
    outer.interpret_line("0 0 0 ARGV >NUMBER 2DROP DROP")
    assert inner.pop_ds_int() == 10


def test_argv_out_of_range():
    """Test ARGV primitive returns 0 0 for out-of-range index."""
    inner = InnerInterpreter()
    inner.argv = ["10"]
    outer = OuterInterpreter(inner)
    outer.interpret_line("1 ARGV")
    assert inner.pop_ds_int() == 0
    assert inner.pop_ds_int() == 0


# ---------------------------------------------------------------------------
# Value-word inlining: a reference to a VARIABLE/CONSTANT/CREATE word inside a
# colon definition should compile to an inline literal push, not a nested-thread
# call. Calls show up in the JIT trace as cs_threads/cs_ips array traffic plus
# two guard_value ops per reference (the dominant cost in the `ary` shootout
# inner loop); inlining the constant push removes that overhead entirely.
# ---------------------------------------------------------------------------

def _compile(*lines):
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    for line in lines:
        outer.interpret_line(line)
    return inner, outer


def test_variable_reference_is_inlined():
    inner, outer = _compile("VARIABLE X", ": GETX X ;")
    thread = outer.dict["GETX"].thread
    x_word = outer.dict["X"]
    # The X reference must NOT compile to a call to the X word.
    assert x_word not in thread.code
    # It must compile to an inline literal push of X's address.
    assert thread.code[0] is outer.wLIT
    outer.interpret_line("X")
    assert thread.lits[0].intval == inner.pop_ds_int()


def test_constant_reference_is_inlined():
    inner, outer = _compile("100 CONSTANT C", ": GETC C ;")
    thread = outer.dict["GETC"].thread
    assert outer.dict["C"] not in thread.code
    assert thread.code[0] is outer.wLIT
    assert thread.lits[0].intval == 100


def test_inlined_value_words_preserve_semantics():
    assert run_and_pop("VARIABLE X 123 X !  : GETX X @ ;  GETX") == 123
    assert run_and_pop("100 CONSTANT C  : DBL C C + ;  DBL") == 200
    assert run_and_pop("VARIABLE Y VARIABLE Z  : DIFF Z Y - ;  DIFF") == CELL_SIZE_BYTES


def test_inlining_does_not_break_tick_or_body():
    # Inlining direct references must not remove the word from the dictionary,
    # so ' (tick) and >BODY still resolve the value-word.
    inner, outer = _compile("VARIABLE X", ": GETX X ;")
    outer.interpret_line("' X >BODY   GETX")
    getx_addr = inner.pop_ds_int()
    body_addr = inner.pop_ds_int()
    assert getx_addr == body_addr


# ---------------------------------------------------------------------------
# Leaf-colon-word inlining: a reference to a *small straight-line* colon word
# (no branches/loops/early-EXIT) is spliced inline at its call site instead of
# emitting a nested-thread call. A call costs cs_threads/cs_ips array
# read+write+null-store plus two guard_value ops in the JIT trace -- the
# dominant remaining cost of the `heap` shootout inner loop (a@/a!/set-rra/...).
# Words containing control flow are NOT inlined (their branch targets are
# absolute indices into their own thread and cannot be relocated).
# ---------------------------------------------------------------------------

def test_leaf_colon_word_is_inlined():
    inner, outer = _compile(
        "VARIABLE B  0 B !",
        ": geta  B @ 1 + ;",          # straight-line: LIT b, @, LIT 1, +, EXIT
        ": useit  geta geta ;",
    )
    geta = outer.dict["GETA"]
    thread = outer.dict["USEIT"].thread
    # geta must NOT be called -- neither as a code word nor a TAILCALL literal.
    assert geta not in thread.code
    assert not any(getattr(l, "word", None) is geta for l in thread.lits)
    # Only the trailing EXIT remains; the body is spliced straight-line.
    assert thread.code[-1] is outer.wEXIT


def test_inlined_leaf_word_semantics():
    assert run_and_pop("VARIABLE B 5 B !  : geta B @ 1 + ;  : useit geta geta + ;  useit") == 12


def test_transitive_inlining():
    # c inlines b which inlined a -- all straight-line.
    assert run_and_pop("VARIABLE B 3 B !  : a B @ ;  : b a 1 + ;  : c b 2 * ;  c") == 8


def test_control_flow_word_not_inlined_but_correct():
    # classify contains IF/ELSE (branch words). It must NOT be inline-spliced
    # (which would corrupt its absolute branch targets), and must still work.
    assert run_and_pop(": classify dup 0 < if drop -1 else drop 1 then ;  : u -5 classify ;  u") == -1
    assert run_and_pop(": classify dup 0 < if drop -1 else drop 1 then ;  : u 5 classify ;  u") == 1


def test_loop_word_not_miscompiled_when_referenced():
    # sumto has a DO loop; referencing it twice must not splice its loop body
    # (whose (LOOP) target is an absolute index into sumto's own thread).
    src = (": sumto  0 swap 0 do i + loop ;  "
           ": twice  5 sumto  5 sumto + ;  twice")
    assert run_and_pop(src) == 20      # sum(0..4)=10, twice=20


def test_inlining_grows_code_buffer_beyond_128():
    # A small leaf referenced many times must grow the compile buffer past 128.
    leaf = ": leaf 1 2 3 4 5 drop drop drop drop ;"   # nets one value (1)
    src = leaf + " : big " + (" ".join(["leaf"] * 20)) + " ;  big"
    assert run_and_pop(src) == 1


def test_rshift_is_logical():
    # Forth RSHIFT is a logical (zero-fill) shift: all-ones shifted right once
    # gives the largest positive cell, not -1.
    assert run_and_pop("-1 1 RSHIFT") == 0x7FFFFFFFFFFFFFFF
    assert run_and_pop("8 2 RSHIFT") == 2
