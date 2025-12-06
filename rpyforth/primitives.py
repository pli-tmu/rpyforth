from rpython.rlib.jit import promote, unroll_safe, dont_look_inside
from rpython.rlib.rfloat import formatd

from rpyforth.objects import (
    BINARY,
    CELL_SIZE_BYTES,
    OCTAL,
    DECIMAL,
    HEX,
    TRUE,
    ZERO,
    W_IntObject,
    W_StringObject,
    W_FloatObject,
    W_WordObject,
    LONG_BIT,
    _small_int_cache,
    SMALL_INT_MIN,
    SMALL_INT_MAX,
)
from rpyforth.inner_interp import jitdriver
from rpyforth.util import digit_to_char


# Internal helpers -----------------------------------------------------------

def _maybe_enter_jit(inner, target_ip, origin_ip, thread):
    """Signal the interpreter back-edge to the JIT when jumping backward."""
    if target_ip < origin_ip:
        jitdriver.can_enter_jit(
            ip=target_ip,
            thread=thread,
            self=inner,
        )


# 0= ( x -- flag )
def prim_ZEROEQUAL(inner, cur, ip):
    """GForth core 2012: flag is true when x equals zero."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = -1 if inner.ds_ints[ptr] == 0 else 0
    return ip


# 0< ( n -- flag )
def prim_ZEROLESS(inner, cur, ip):
    """GForth core 2012: flag is true when n is strictly negative."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = -1 if inner.ds_ints[ptr] < 0 else 0
    return ip


# 0> ( n -- flag )
def prim_ZEROGREATER(inner, cur, ip):
    """GForth core 2012: flag is true when n is strictly positive."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = -1 if inner.ds_ints[ptr] > 0 else 0
    return ip


# > ( n1 n2 -- flag )
def prim_GREATER(inner, cur, ip):
    """GForth core 2012: flag is true when n1 is greater than n2."""
    ptr = inner.ds_ptr_ints
    n2 = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = -1 if n1 > n2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip

# < ( n1 n2 -- flag )
def prim_LESS(inner, cur, ip):
    """GForth core 2012: flag is true when n1 is less than n2."""
    ptr = inner.ds_ptr_ints
    n2 = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = -1 if n1 < n2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip


# 0<> ( n -- flag )
def prim_ZERONOTEQUAL(inner, cur, ip):
    """GForth core 2012: flag is true when n is non-zero."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = -1 if inner.ds_ints[ptr] != 0 else 0
    return ip

# def U< (n1 n2 -- flag )
def prim_U_LESS(inner, cur, ip):
    """GForth core 2012: flag is true if and only if u1 is less than u2."""
    ptr = inner.ds_ptr_ints
    n2 = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    # Compute mask inline to avoid RPython long literal issue
    mask = (1 << LONG_BIT) - 1
    u1 = n1 & mask
    u2 = n2 & mask
    inner.ds_ints[ptr - 2] = -1 if u1 < u2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip

# DUP ( x -- x x )
def prim_DUP(inner, cur, ip):
    """GForth core 2012: duplicate x, leaving two copies on the stack."""
    a = inner.peek_ds_int(0)
    inner.push_ds_int(a)
    return ip


# 2DUP ( x1 x2 -- x1 x2 x1 x2 )
def prim_2DUP(inner, cur, ip):
    """GForth core 2012: duplicate cell pair x1 x2."""
    b = inner.peek_ds_int(0)
    a = inner.peek_ds_int(1)
    inner.push_ds_int(a)
    inner.push_ds_int(b)
    return ip


# ?DUP ( x -- 0 | x x )
def prim_QUESTIONDUP(inner, cur, ip):
    """GForth core 2012: duplicate x if it is non-zero."""
    a = inner.peek_ds_int(0)
    if a != 0:
        inner.push_ds_int(a)
    return ip


# DROP ( x -- )
def prim_DROP(inner, cur, ip):
    """GForth core 2012: discard the top stack item."""
    inner.ds_ptr_ints -= 1
    return ip


# NIP ( x1 x2 -- x2 )
def prim_NIP(inner, cur, ip):
    """GForth core 2012: discard the second stack item."""
    x2 = inner.peek_ds_int(0)
    inner.poke_ds_int(1, x2)
    inner.ds_ptr_ints -= 1
    return ip


# 2DROP ( x1 x2 -- )
def prim_2DROP(inner, cur, ip):
    """GForth core 2012: discard the top two stack items."""
    inner.ds_ptr_ints -= 2
    return ip

# SWAP ( x1 x2 -- x2 x1 )
def prim_SWAP(inner, cur, ip):
    """GForth core 2012: exchange the top two stack items."""
    a, b = inner.top2_ds_int()
    inner.push_ds_int(b)
    inner.push_ds_int(a)
    return ip


# 2SWAP ( x1 x2 x3 x4 -- x3 x4 x1 x2 )
def prim_2SWAP(inner, cur, ip):
    """GForth core 2012: exchange the top two stack items."""
    d = inner.peek_ds_int(0)  # x4 (top)
    c = inner.peek_ds_int(1)  # x3
    b = inner.peek_ds_int(2)  # x2
    a = inner.peek_ds_int(3)  # x1 (bottom)
    inner.poke_ds_int(3, c)
    inner.poke_ds_int(2, d)
    inner.poke_ds_int(1, a)
    inner.poke_ds_int(0, b)
    return ip


# OVER ( x1 x2 -- x1 x2 x1 )
def prim_OVER(inner, cur, ip):
    """GForth core 2012: copy the second stack item to the top."""
    a = inner.peek_ds_int(1)
    inner.push_ds_int(a)
    return ip


# 2OVER ( x1 x2 x3 x4 -- x1 x2 x3 x4 x1 x2 )
def prim_2OVER(inner, cur, ip):
    """GForth core 2012: copy cell pair x1 x2 to the top of the stack."""
    a = inner.peek_ds_int(3)
    b = inner.peek_ds_int(2)
    inner.push_ds_int(a)
    inner.push_ds_int(b)
    return ip


# ROT ( x1 x2 x3 -- x2 x3 x1 )
def prim_ROT(inner, cur, ip):
    """GForth core 2012: rotate the top three stack items."""
    c = inner.peek_ds_int(0)  # x3 (top)
    b = inner.peek_ds_int(1)  # x2 (middle)
    a = inner.peek_ds_int(2)  # x1 (bottom)
    inner.poke_ds_int(2, b)   # x2 goes to bottom
    inner.poke_ds_int(1, c)   # x3 goes to middle
    inner.poke_ds_int(0, a)   # x1 goes to top
    return ip


# -ROT ( x1 x2 x3 -- x3 x1 x2 )
def prim_NROT(inner, cur, ip):
    """Inverse of ROT: rotate in the opposite direction."""
    c = inner.peek_ds_int(0)  # x3 (top)
    b = inner.peek_ds_int(1)  # x2 (middle)
    a = inner.peek_ds_int(2)  # x1 (bottom)
    inner.poke_ds_int(2, c)   # x3 goes to bottom
    inner.poke_ds_int(1, a)   # x1 goes to middle
    inner.poke_ds_int(0, b)   # x2 goes to top
    return ip


# MAX ( n1 n2 -- n3 )
def prim_MAX(inner, cur, ip):
    """GForth core 2012: n3 is the greater of n1 and n2."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = b if a < b else a
    inner.ds_ptr_ints = ptr - 1
    return ip


# MIN ( n1 n2 -- n3 )
def prim_MIN(inner, cur, ip):
    """GForth core 2012: n3 is the lesser of n1 and n2."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a if a < b else b
    inner.ds_ptr_ints = ptr - 1
    return ip


# DEPTH ( -- +n )
def prim_DEPTH(inner, cur, ip):
    """GForth core 2012: +n is the number of single-cell values contained in the data stack."""
    inner.push_ds_int(inner.ds_ptr_ints)
    return ip


# RSHIFT ( n1 u -- n2 )
def prim_RSHIFT(inner, cur, ip):
    """GForth core 2012: perform a logical right shift of u bit-places on n1, giving n2."""
    ptr = inner.ds_ptr_ints
    u = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = n1 >> u
    inner.ds_ptr_ints = ptr - 1
    return ip


# LSHIFT ( n1 u -- n2 )
def prim_LSHIFT(inner, cur, ip):
    """GForth core 2012: perform a logical left shift of u bit-places on n1, giving n2."""
    ptr = inner.ds_ptr_ints
    u = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = n1 << u
    inner.ds_ptr_ints = ptr - 1
    return ip

# S>D ( n -- d )
def prim_S_TO_D(inner, cur, ip):
    """GForth core 2012: convert tne number n to double-cell number d."""
    a = inner.pop_ds_int()
    inner.push_ds_int(a)
    if a >= 0:
        inner.push_ds_int(0)
    else:
        inner.push_ds_int(-1)
    return ip


# D+ ( d1 d2 -- d3 )
def prim_D_PLUS(inner, cur, ip):
    """GForth double 2012: add d1 to d2, giving the sum d3."""
    # Stack: d1.lo d1.hi d2.lo d2.hi (d2.hi on top)
    d2_hi = inner.pop_ds_int()
    d2_lo = inner.pop_ds_int()
    d1_hi = inner.pop_ds_int()
    d1_lo = inner.pop_ds_int()
    BIT_MASK = (1 << LONG_BIT) - 1
    # Combine into 128-bit values (using Python's arbitrary precision)
    d1 = d1_lo + (d1_hi << LONG_BIT)
    d2 = d2_lo + (d2_hi << LONG_BIT)
    result = d1 + d2
    # Split back into low and high cells
    result_lo = result & BIT_MASK
    result_hi = (result >> LONG_BIT) & BIT_MASK
    inner.push_ds_int(result_lo)
    inner.push_ds_int(result_hi)
    return ip


# D- ( d1 d2 -- d3 )
def prim_D_MINUS(inner, cur, ip):
    """GForth double 2012: subtract d2 from d1, giving the difference d3."""
    # Stack: d1.lo d1.hi d2.lo d2.hi (d2.hi on top)
    d2_hi = inner.pop_ds_int()
    d2_lo = inner.pop_ds_int()
    d1_hi = inner.pop_ds_int()
    d1_lo = inner.pop_ds_int()
    BIT_MASK = (1 << LONG_BIT) - 1
    # Combine into 128-bit values (using Python's arbitrary precision)
    d1 = d1_lo + (d1_hi << LONG_BIT)
    d2 = d2_lo + (d2_hi << LONG_BIT)
    result = d1 - d2
    # Split back into low and high cells
    result_lo = result & BIT_MASK
    result_hi = (result >> LONG_BIT) & BIT_MASK
    inner.push_ds_int(result_lo)
    inner.push_ds_int(result_hi)
    return ip


# D. ( d -- )
def prim_D_DOT(inner, cur, ip):
    """GForth double 2012: display d according to current BASE."""
    # Stack: d.lo d.hi (d.hi on top)
    d_hi = inner.pop_ds_int()
    d_lo = inner.pop_ds_int()
    # Combine into full value
    d = d_lo + (d_hi << LONG_BIT)
    stdout = inner.stdout
    stdout.write(str(d))
    stdout.write(' ')
    return ip


# BL ( -- char )
def prim_BL(inner, cur, ip):
    """GForth core 2012: char is the character value of a space."""
    inner.push_ds_int(ord(' '))
    return ip

# 2* ( x1 -- x2 )
def prim_2STAR(inner, cur, ip):
    """GForth core 2012: x2 is the result of shifting x1 one bit toward the most-significant bit."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = inner.ds_ints[ptr] << 1
    return ip

# 2/ ( x1 -- x2 )
def prim_2SLASH(inner, cur, ip):
    """GForth core 2012: x2 is the result of shifting x1 one bit right towards the least-significant bit."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = inner.ds_ints[ptr] >> 1
    return ip

# Arithmetic

def int_shortcut(intval):
    if SMALL_INT_MIN <= intval < SMALL_INT_MAX:
        return _small_int_cache[intval - SMALL_INT_MIN]
    else:
        return W_IntObject(intval)

# + ( n1 n2 -- n3 )
def prim_ADD(inner, cur, ip):
    """GForth core 2012: add n1 and n2, leaving their sum."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a + b
    inner.ds_ptr_ints = ptr - 1
    return ip


# - ( n1 n2 -- n3 )
def prim_SUB(inner, cur, ip):
    """GForth core 2012: subtract n2 from n1, leaving the difference."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a - b
    inner.ds_ptr_ints = ptr - 1
    return ip


# * ( n1 n2 -- n3 )
def prim_MUL(inner, cur, ip):
    """GForth core 2012: multiply n1 by n2, leaving the product."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a * b
    inner.ds_ptr_ints = ptr - 1
    return ip

# / ( n1 n2 -- n3 )
def prim_DIV(inner, cur, ip):
    """GForth core 2012: divide n1 by n2, giving the single-cell quotient n3."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    assert b != 0, "Division by zero"
    inner.ds_ints[ptr - 2] = a // b
    inner.ds_ptr_ints = ptr - 1
    return ip

# */ ( n1 n2 n3 -- n4 )
def prim_MUL_SLASH(inner, cur, ip):
    """GForth core 2012: multiply n1 by n2 producing the intermediate double-cell result d. divide d by n3 giving the single-cell quotient n4."""
    n3 = inner.pop_ds_int()
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    d = n1 * n2
    assert n3 != 0, "Division by zero"
    result = d // n3
    assert (result >> LONG_BIT) == 0 or (result >> LONG_BIT) == -1, "Overflow in */"
    inner.push_ds_int(result)
    return ip

# /MOD ( n1 n2 -- n3 n4 )
def prim_DIV_MOD(inner, cur, ip):
    """GForth core 2012: divide n1 by n2, giving the single-cell remainder n3 and the single-cell quotient n4."""
    a, b = inner.top2_ds_int()
    assert b != 0, "Division by zero"
    inner.push_ds_int(a % b)
    inner.push_ds_int(a // b)
    return ip

# */MOD ( n1 n2 n3 -- n4 n5 )
def prim_MUL_DIV_MOD(inner, cur, ip):
    """GForth core 2012: multiply n1 by n2 producing the intermediate double-cell result d. divide d by n3 giving the single-cell remainder n4 and the single-cell quotient n5."""
    n3 = inner.pop_ds_int()
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    d = n1 * n2
    assert n3 != 0, "Division by zero"
    q = d // n3
    assert (q >> LONG_BIT) == 0 or (q >> LONG_BIT) == -1, "Overflow in */mod"
    inner.push_ds_int(d % n3)
    inner.push_ds_int(q)
    return ip

# ABS ( n -- u )
def prim_ABS(inner, cur, ip):
    """GForth core 2012: u is the absolute value of n."""
    ptr = inner.ds_ptr_ints - 1
    a = inner.ds_ints[ptr]
    if a < 0:
        inner.ds_ints[ptr] = -a
    return ip


# NEGATE ( n1 -- n2 )
def prim_NEGATE(inner, cur, ip):
    """GForth core 2012: negate n1, giving its arithmetic inverse n2."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = -inner.ds_ints[ptr]
    return ip


# MOD ( n1 n2 -- n3 )
def prim_MOD(inner, cur, ip):
    """GForth core 2012: divide n1 by n2, giving the single-cell remainder n3."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a % b
    inner.ds_ptr_ints = ptr - 1
    return ip


# /MOD ( n1 n2 -- n3 n4 )
def prim_DIVMOD(inner, cur, ip):
    """GForth core 2012: divide n1 by n2, giving remainder n3 and quotient n4."""
    a, b = inner.top2_ds_int()
    # Symmetric division
    q = int(a / b)
    r = a - (q * b)
    inner.push_ds_int(r)
    inner.push_ds_int(q)
    return ip


# */ ( n1 n2 n3 -- n4 )
def prim_STARSLASH(inner, cur, ip):
    """GForth core 2012: multiply n1 by n2 producing intermediate double-cell result, divide by n3."""
    n3 = inner.pop_ds_int()
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    # Use intermediate double-cell precision
    product = n1 * n2
    result = int(product / n3)
    inner.push_ds_int(result)
    return ip


# */MOD ( n1 n2 n3 -- n4 n5 )
def prim_STARSLASHMOD(inner, cur, ip):
    """GForth core 2012: multiply n1 by n2, divide by n3, giving remainder n4 and quotient n5."""
    n3 = inner.pop_ds_int()
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    # Use intermediate double-cell precision
    product = n1 * n2
    q = int(product / n3)
    r = product - (q * n3)
    inner.push_ds_int(r)
    inner.push_ds_int(q)
    return ip


# FM/MOD ( d n1 -- n2 n3 )
def prim_FMSLASHMOD(inner, cur, ip):
    """GForth core 2012: floored division of double d by n1, giving remainder n2 and quotient n3."""
    n1 = inner.pop_ds_int()
    d_hi = inner.pop_ds_int()
    d_lo = inner.pop_ds_int()

    # Reconstruct double-cell dividend (simplified: use d_lo for single-cell)
    dividend = d_lo
    divisor = n1

    # Floored division: quotient rounded toward negative infinity
    if divisor == 0:
        inner.push_ds_int(0)
        inner.push_ds_int(0)
        return ip

    # Python's // is floored division
    q = dividend // divisor
    r = dividend - (q * divisor)

    inner.push_ds_int(r)
    inner.push_ds_int(q)
    return ip


# SM/REM ( d n1 -- n2 n3 )
def prim_SMSLASHREM(inner, cur, ip):
    """GForth core 2012: symmetric division of double d by n1, giving remainder n2 and quotient n3."""
    n1 = inner.pop_ds_int()
    d_hi = inner.pop_ds_int()
    d_lo = inner.pop_ds_int()

    # Reconstruct double-cell dividend (simplified: use d_lo for single-cell)
    dividend = d_lo
    divisor = n1

    if divisor == 0:
        inner.push_ds_int(0)
        inner.push_ds_int(0)
        return ip

    # Symmetric division: quotient rounded toward zero
    q = int(dividend / divisor)
    r = dividend - (q * divisor)

    inner.push_ds_int(r)
    inner.push_ds_int(q)
    return ip


# UM/MOD ( ud u1 -- u2 u3 )
def prim_UMSLASHMOD(inner, cur, ip):
    """GForth core 2012: unsigned division of double ud by u1, giving remainder u2 and quotient u3."""
    u1 = inner.pop_ds_int()
    ud_hi = inner.pop_ds_int()
    ud_lo = inner.pop_ds_int()

    # For unsigned division, treat values as unsigned
    # Reconstruct the double-cell unsigned value
    BIT_MASK = (1 << LONG_BIT) - 1
    lo = ud_lo & BIT_MASK
    hi = ud_hi & BIT_MASK
    dividend = (hi << LONG_BIT) | lo
    divisor = u1 & BIT_MASK

    if divisor == 0:
        inner.push_ds_int(0)
        inner.push_ds_int(0)
        return ip

    q = dividend // divisor
    r = dividend % divisor

    inner.push_ds_int(r)
    inner.push_ds_int(q)
    return ip


# 1+ ( n1 -- n2 )
def prim_INC(inner, cur, ip):
    """GForth core 2012: add one to n1."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = inner.ds_ints[ptr] + 1
    return ip


# 1- ( n1 -- n2 )
def prim_DEC(inner, cur, ip):
    """GForth core 2012: subtract one from n1."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = inner.ds_ints[ptr] - 1
    return ip


# M* ( n1 n2 -- d)
def prim_MUL_STAR(inner, cur, ip):
    """GForth core 2012: d is the signed product of n1 times n2."""
    a, b = inner.top2_ds_int()
    c = a * b    #c is 128bits

    BIT_MASK = (1 << LONG_BIT) - 1   #111...11 64bits
    SIGN_BIT = 1 << (LONG_BIT - 1)  #100...00 64bits

    low = c & BIT_MASK    # get c's low 64bits
    #high 64bits: 0s, low 64bits: c's low 64bits,total 128bits

    if low & SIGN_BIT:  # if highest of c's low bits is 1
    #because highest bit of 128bits is 0, conversion is required

        low = low - (1 << LONG_BIT)  # convert to negative number

    high = c >> LONG_BIT # get c's high 64bits
    #(ex,LONG_BIT = 4) if c = 0100 0000, high = 0000 0100 = c's high 4bits : if c = 1100 1000, high = 1111 1100 = c's high 4bits

    inner.push_ds_int(low)
    inner.push_ds_int(high)

    return ip

# UM* ( n1 n2 -- d)
def prim_U_MUL_STAR(inner, cur, ip):
    """GForth core 2012: multiply u1 by u2, giving the unsigned double-cell product ud."""
    a, b = inner.top2_ds_int()
    c = a * b    #c is 128bits

    BIT_MASK = (1 << LONG_BIT) - 1   #111...11 64bits

    low = c & BIT_MASK    # get c's low 64bits
    #high 64bits: 0s, low 64bits: c's low 64bits,total 128bits

    high = c >> LONG_BIT # get c's high 64bits
    #(ex,LONG_BIT = 4) if c = 0100 0000, high = 0000 0100 = c's high 4bits : if c = 1100 1000, high = 1111 1100 = c's high 4bits

    inner.push_ds_int(low)
    inner.push_ds_int(high)

    return ip

# AND ( x1 x2 -- x3 )
def prim_AND(inner, cur, ip):
    """GForth core 2012: x3 is the bit-by-bit logical "and" of x1 with x2."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a & b
    inner.ds_ptr_ints = ptr - 1
    return ip


# OR ( x1 x2 -- x3 )
def prim_OR(inner, cur, ip):
    """GForth core 2012: x3 is the bit-by-bit inclusive-or of x1 with x2."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a | b
    inner.ds_ptr_ints = ptr - 1
    return ip

# XOR ( x1 x2 -- x3 )
def prim_XOR(inner, cur, ip):
    """GForth core 2012: x3 is the bit-by-bit exclusive-or of x1 with x2."""
    ptr = inner.ds_ptr_ints
    b = inner.ds_ints[ptr - 1]
    a = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = a ^ b
    inner.ds_ptr_ints = ptr - 1
    return ip


# FM/MOD ( d1 n1 -- n2 n3 )
def prim_FM_DIV_MOD(inner, cur, ip):
    """GForth core 2012: divide d1 by n1, giving the floored quotient n3 and the remainder n2."""
    a = inner.pop_ds_int()
    b = inner.pop_ds_int() # d1's high 64bits
    c = inner.pop_ds_int() # d1's low 64bits
    BIT_MASK = (1 << LONG_BIT) - 1   #111...11 64bits
    d = (b << LONG_BIT) | (c & BIT_MASK) #d is 128bits
    assert a != 0, "Division by zero"
    e = d // a #e is 64bits
    assert (e >> LONG_BIT) == 0 or (e >> LONG_BIT) == -1, "Overflow in fm/mod"
    inner.push_ds_int(d % a)
    inner.push_ds_int(e)
    return ip

# UM/MOD ( ud u1 -- u2 u3 )
def prim_UM_DIV_MOD(inner, cur, ip):
    """GForth core 2012: divide ud by u1, giving the quotient u3 and the remainder u2."""
    a = inner.pop_ds_int()
    b = inner.pop_ds_int() # ud's high 64bits
    c = inner.pop_ds_int() # ud's low 64bits
    BIT_MASK = (1 << LONG_BIT) - 1   #111...11 64bits
    SIGN_BIT = 1 << (LONG_BIT - 1)  #100...00 64bits
    d = (b << LONG_BIT) | (c & BIT_MASK) #d is 128bits
    assert a != 0, "Division by zero"
    e = d // a #e is 64bits (quotient)
    f = d % a #f is 64bits (remainder)
    if e & SIGN_BIT:  # if highest of e's bits is 1
        e = e - (1 << LONG_BIT)  # convert to negative number
    if f & SIGN_BIT:  # if highest of f's bits is 1
        f = f - (1 << LONG_BIT)  # convert to negative number
    assert (e >> LONG_BIT) == 0 or (e >> LONG_BIT) == -1, "Overflow in um/mod"
    inner.push_ds_int(f)
    inner.push_ds_int(e)
    return ip

# SM/REM ( d1 n1 -- n2 n3 )
def prim_SM_DIV_REM(inner, cur, ip):
    """GForth core 2012: divide d1 by n1, giving the symmetric quotient n3 and the remainder n2."""
    a = inner.pop_ds_int()
    b = inner.pop_ds_int() # d1's high 64bits
    c = inner.pop_ds_int() # d1's low 64bits
    BIT_MASK = (1 << LONG_BIT) - 1   #111...11 64bits
    d = (b << LONG_BIT) | (c & BIT_MASK) #d is 128bits
    assert a != 0, "Division by zero"
    a_abs = abs(a)
    d_abs = abs(d)
    e = d_abs // a_abs
    f = d_abs % a_abs
    if (d < 0) ^ (a < 0):  # if signs of d and a are different
        e = -e
    if d < 0:
        f = -f
    inner.push_ds_int(f)
    inner.push_ds_int(e)
    return ip


# INVERT ( x1 -- x2 )
def prim_INVERT(inner, cur, ip):
    """GForth core 2012: invert all bits of x1, giving x2 (one's complement)."""
    ptr = inner.ds_ptr_ints - 1
    inner.ds_ints[ptr] = ~inner.ds_ints[ptr]
    return ip


# U< ( u1 u2 -- flag )
def prim_ULESS(inner, cur, ip):
    """GForth core 2012: flag is true if and only if u1 is less than u2 (unsigned comparison)."""
    ptr = inner.ds_ptr_ints
    u2 = inner.ds_ints[ptr - 1]
    u1 = inner.ds_ints[ptr - 2]
    # Compute mask inline to avoid RPython long literal issue
    mask = (1 << LONG_BIT) - 1
    val1 = u1 & mask
    val2 = u2 & mask
    inner.ds_ints[ptr - 2] = -1 if val1 < val2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip

# memory management


# ! ( x addr -- )
def prim_STORE(inner, cur, ip):
    """GForth core 2012: store x at cell address addr."""
    addr = inner.pop_ds_int()
    val = inner.pop_ds_int()
    inner.cell_store(addr, val)
    return ip


# ! ( x1 x2 a-addr -- )
def prim_2STORE(inner, cur, ip):
    """
    Store the cell pair x1 x2 at a-addr,
    with x2 at a-addr and x1 at the next consecutive cell.
    It is equivalent to the sequence SWAP OVER ! CELL+ !.
    """
    addr = inner.pop_ds_int()
    x2 = inner.pop_ds_int()
    x1 = inner.pop_ds_int()
    inner.cell_2store(addr, x1, x2)
    return ip

# @ ( addr -- x )
def prim_FETCH(inner, cur, ip):
    """GForth core 2012: fetch the cell contents at addr."""
    addr = inner.pop_ds_int()
    w_x = inner.cell_fetch(addr)
    assert isinstance(w_x, W_IntObject)
    inner.push_ds_int(w_x.intval)
    return ip


# ( -- n )
def prim_CELL(inner, cur, ip):
    """push the size of one cell in address units."""
    inner.push_ds_int(CELL_SIZE_BYTES)
    return ip


# FLOAT ( -- n )
def prim_FLOAT(inner, cur, ip):
    """GForth floating 2012: return the size of one float in address units."""
    # In our implementation, floats are stored as W_FloatObject which uses 8 bytes
    inner.push_ds_int(8)
    return ip


# FLOATS ( n1 -- n2 )
def prim_FLOATS(inner, cur, ip):
    """GForth floating 2012: convert  float count to address units."""
    n = inner.pop_ds_int()
    # Each float is 8 bytes
    inner.push_ds_int(n * 8)
    return ip


# ( n -- n )
def prim_CELLPLUS(inner, cur, ip):
    """GForth core 2012: add one cell to an address."""
    addr = inner.pop_ds_int()
    inner.push_ds_int(addr + CELL_SIZE_BYTES)
    return ip


# ( n -- n * cell_size )
def prim_CELLS(inner, cur, ip):
    """GForth core 2012: convert a cell count to address units."""
    count = inner.pop_ds_int()
    inner.push_ds_int(count * CELL_SIZE_BYTES)
    return ip


# IF THEN ELSE


# 0BRANCH ( flag -- )
def prim_0BRANCH(inner, cur, ip):
    """GForth core 2012: branch to target when flag is zero."""
    origin_ip = ip - 1
    x = inner.pop_ds_int()
    if x == 0:
        w_target = promote(cur.lits[origin_ip])
        assert isinstance(w_target, W_IntObject)
        target_ip = w_target.intval
        ip = target_ip
        _maybe_enter_jit(inner, target_ip, origin_ip, cur)
    return ip


# BRANCH ( -- )
def prim_BRANCH(inner, cur, ip):
    """GForth core 2012: branch unconditionally to the target."""
    origin_ip = ip - 1
    target = promote(cur.lits[origin_ip])
    assert isinstance(target, W_IntObject)
    target_ip = target.intval
    ip = target_ip
    _maybe_enter_jit(inner, target_ip, origin_ip, cur)
    return ip


# Loop control primitives

# (DO) ( limit start -- )
def prim_DO_RUNTIME(inner, cur, ip):
    """Runtime for DO: pop limit and start from data stack, push to loop stack."""
    ds_ptr = inner.ds_ptr_ints
    start = inner.ds_ints[ds_ptr - 1]
    limit = inner.ds_ints[ds_ptr - 2]
    inner.ds_ptr_ints = ds_ptr - 2
    # Direct return stack push
    rs_ptr = inner.rs_ptr
    inner.rs[rs_ptr] = limit
    inner.rs[rs_ptr + 1] = start
    inner.rs_ptr = rs_ptr + 2
    return ip


# (LOOP) ( -- ) ( R: limit counter -- limit counter+1 | )
@unroll_safe
def prim_LOOP_RUNTIME(inner, cur, ip):
    rs_ptr = inner.rs_ptr
    counter_val = inner.rs[rs_ptr - 1]
    limit_val = promote(inner.rs[rs_ptr - 2])
    new_counter_val = counter_val + 1

    if new_counter_val < limit_val:
        # Continue loop: update counter in place (direct write, no function call)
        inner.rs[rs_ptr - 1] = new_counter_val
        origin_ip = ip - 1
        target = promote(cur.lits[origin_ip])
        assert isinstance(target, W_IntObject)
        target_ip = target.intval
        ip = target_ip
        _maybe_enter_jit(inner, target_ip, origin_ip, cur)
    else:
        # Loop done: pop 2 cells from return stack
        inner.rs_ptr = rs_ptr - 2
    return ip

# (+LOOP) ( n -- ) ( R: limit counter -- limit counter+n | )
@unroll_safe
def prim_PLUSLOOP_RUNTIME(inner, cur, ip):
    """Runtime for +LOOP: increment counter by n and conditionally branch back."""
    ds_ptr = inner.ds_ptr_ints
    inc_val = inner.ds_ints[ds_ptr - 1]
    inner.ds_ptr_ints = ds_ptr - 1

    # Direct return stack access
    rs_ptr = inner.rs_ptr
    counter_val = inner.rs[rs_ptr - 1]
    limit_val = inner.rs[rs_ptr - 2]
    new_counter_val = counter_val + inc_val

    # Check if loop should continue based on crossing the boundary
    # For positive increment: continue while new_counter < limit
    # For negative increment: continue while new_counter >= limit
    continue_loop = False
    if inc_val >= 0:
        # Positive increment: continue if we haven't reached limit
        if counter_val < limit_val and new_counter_val < limit_val:
            continue_loop = True
    else:
        # Negative increment: continue if we haven't gone below limit
        if counter_val >= limit_val and new_counter_val >= limit_val:
            continue_loop = True

    if continue_loop:
        # Continue loop: update counter in place (direct write)
        inner.rs[rs_ptr - 1] = new_counter_val
        origin_ip = ip - 1
        target = promote(cur.lits[origin_ip])
        assert isinstance(target, W_IntObject)
        target_ip = target.intval
        ip = target_ip
        _maybe_enter_jit(inner, target_ip, origin_ip, cur)
    else:
        # Loop done: pop 2 cells from return stack
        inner.rs_ptr = rs_ptr - 2
    return ip


# UNLOOP ( -- ) ( R: limit counter -- )
def prim_UNLOOP(inner, cur, ip):
    """GForth core 2012: discard loop parameters from return stack."""
    inner.rs_ptr -= 2
    return ip


# LEAVE ( -- ) ( R: limit counter -- )
def prim_LEAVE(inner, cur, ip):
    """Exit the current loop by cleaning up return stack and jumping to end."""
    inner.rs_ptr -= 2
    target = promote(cur.lits[ip - 1])
    assert isinstance(target, W_IntObject)
    ip = target.intval
    return ip

# I ( -- n ) ( R: limit counter -- limit counter )
@unroll_safe
def prim_I(inner, cur, ip):
    """Get the current loop counter (innermost loop)."""
    rs_ptr = inner.rs_ptr
    counter_val = inner.rs[rs_ptr - 1]
    inner.push_ds_int(counter_val)
    return ip


# J ( -- n ) ( R: limit1 counter1 limit2 counter2 -- limit1 counter1 limit2 counter2 )
@unroll_safe
def prim_J(inner, cur, ip):
    """Get the outer loop counter (second innermost loop)."""
    rs_ptr = inner.rs_ptr
    counter_val = inner.rs[rs_ptr - 3]  # skip inner counter + inner limit
    inner.push_ds_int(counter_val)
    return ip


# BASE


# BASE@ ( -- u )
def prim_BASE_FETCH(inner, cur, ip):
    """GForth core 2012: return the current conversion base."""
    inner.push_ds_int(inner.base)
    return ip


# BASE! ( u -- )
def prim_BASE_STORE(inner, cur, ip):
    """GForth core 2012: set the conversion base to u."""
    u = inner.pop_ds_int()
    inner.base = u
    return ip


# DECIMAL ( -- )
def prim_DECIMAL(inner, cur, ip):
    """GForth core 2012: set BASE to decimal (radix 10)."""
    inner.base = 10
    return ip


# HEX ( -- )
def prim_HEX(inner, cur, ip):
    """GForth core 2012: set BASE to hexadecimal (radix 16)."""
    inner.base = 16
    return ip


# OCTAL ( -- )
def prim_OCTAL(inner, cur, ip):
    """GForth core 2012: set BASE to octal (radix 8)."""
    inner.base = 8
    return ip


# BINARY ( -- )
def prim_BINARY(inner, cur, ip):
    """GForth core 2012: set BASE to binary (radix 2)."""
    inner.base = 2
    return ip


# <# ( -- )
def prim_LESSNUM(inner, cur, ip):
    """GForth core 2012: begin pictured numeric output conversion."""
    inner._pno_active = True
    inner._pno_buf = []
    return ip


# # ( ud1 -- ud2 )
def prim_NUMSIGN(inner, cur, ip):
    """GForth core 2012: extract one digit during pictured numeric output."""
    if not inner._pno_active:
        inner.print_str(W_StringObject("# outside <# #>"))
        return ip
    x = inner.pop_ds_int()
    base = inner.base
    q = x // base
    r = x % base
    inner._pno_buf.insert(0, digit_to_char(r))
    inner.push_ds_int(q)
    return ip


# #S ( ud -- ud )
@unroll_safe
def prim_NUMSIGN_S(inner, cur, ip):
    """GForth core 2012: convert all remaining digits during pictured numeric output."""
    if not inner._pno_active:
        inner.print_str(W_StringObject("#S outside <# #>"))
        return ip

    # Pop double-cell number (d.lo d.hi) where d.hi is on top
    hi = inner.pop_ds_int()
    lo = inner.pop_ds_int()

    # For simplified implementation, use the low-order cell
    # (assumes the number fits in single cell)

    base = inner.base
    # Convert all remaining digits
    if lo == 0:
        # At least one digit for zero
        inner._pno_buf.insert(0, digit_to_char(0))
    else:
        while lo > 0:
            q = lo // base
            r = lo % base
            inner._pno_buf.insert(0, digit_to_char(r))
            lo = q

    # Push double-cell zero (0 0)
    inner.push_ds_int(0)
    inner.push_ds_int(0)
    return ip


# HOLD ( char -- )
def prim_HOLD(inner, cur, ip):
    """GForth core 2012: insert character into pictured numeric output buffer."""
    if not inner._pno_active:
        inner.print_str(W_StringObject("HOLD outside <# #>"))
        return ip
    ch = inner.pop_ds_int()
    inner._pno_buf.insert(0, chr(ch))
    return ip


# SIGN ( n -- )
def prim_SIGN(inner, cur, ip):
    """GForth core 2012: add a minus sign to pictured numeric output if n is negative."""
    if not inner._pno_active:
        inner.print_str(W_StringObject("SIGN outside <# #>"))
        return ip
    n = inner.pop_ds_int()
    if n < 0:
        # Append to put the sign at the end (left side of the final string)
        inner._pno_buf.append('-')
    return ip


# #> ( xd -- c-addr u )
def prim_NUMGREATER(inner, cur, ip):
    """GForth core 2012: finish pictured numeric output and deliver the string."""
    if not inner._pno_active:
        inner.print_str(W_StringObject("#> outside <# #>"))
        return ip
    _ = inner.pop_ds_int()
    s = "".join(inner._pno_buf)
    inner._pno_active = False
    inner.push_ds(W_StringObject(s))
    return ip


# TYPE ( c-addr u -- )
@dont_look_inside
def prim_TYPE(inner, cur, ip):
    """GForth core 2012: display the character string."""
    w_s = inner.pop_ds()
    inner.print_str(w_s)
    return ip


# I/O


# . ( n -- )
@dont_look_inside
def prim_DOT(inner, cur, ip):
    """GForth core 2012: display n according to current BASE."""
    x = inner.pop_ds_int()
    stdout = inner.stdout
    stdout.write(str(x))
    stdout.write(' ')
    #stdout.flush()
    return ip

@dont_look_inside
def prim_U_DOT(inner, cur, ip):
    """GForth core 2012: display u in field format according to current BASE."""
    x = inner.pop_ds_int()
    stdout = inner.stdout
    BIT_MASK = (1 << LONG_BIT) - 1  #111...11 64bits
    u_value = x & BIT_MASK
    stdout.write(str(u_value))
    stdout.write(' ')
    #stdout.flush()
    return ip


# EMIT ( char -- )
@dont_look_inside
def prim_EMIT(inner, cur, ip):
    """GForth core 2012: display character with char code."""
    x = inner.pop_ds_int()
    stdout = inner.stdout
    stdout.write(chr(x))
    stdout.flush()
    return ip


# SPACE ( -- )
@dont_look_inside
def prim_SPACE(inner, cur, ip):
    """GForth core 2012: display one space."""
    stdout = inner.stdout
    stdout.write(' ')
    return ip


# SPACES ( n -- )
@dont_look_inside
def prim_SPACES(inner, cur, ip):
    """GForth core 2012: display n spaces."""
    count = inner.pop_ds_int()
    if count > 0:
        stdout = inner.stdout
        for i in range(count):
            stdout.write(' ')
    return ip


# CR ( -- )
@dont_look_inside
def prim_CR(inner, cur, ip):
    """GForth core 2012: cause subsequent output to appear at the beginning of the next line."""
    stdout = inner.stdout
    stdout.write('\n')
    stdout.flush()
    return ip


# U. ( u -- )
@dont_look_inside
def prim_UDOT(inner, cur, ip):
    """GForth core 2012: display u as unsigned according to current BASE."""
    x = inner.pop_ds_int()
    # Treat as unsigned
    BIT_MASK = (1 << LONG_BIT) - 1
    val = x & BIT_MASK
    stdout = inner.stdout
    stdout.write(str(val))
    stdout.write(' ')
    return ip


# KEY ( -- char )
@dont_look_inside
def prim_KEY(inner, cur, ip):
    """GForth core 2012: receive one character from input device."""
    stdin = inner.stdin
    ch = stdin.read(1)
    if len(ch) > 0:
        # Index into string to get a single character for ord()
        inner.push_ds_int(ord(ch[0]))
    else:
        inner.push_ds_int(0)
    return ip


# ACCEPT ( c-addr +n1 -- +n2 )
@dont_look_inside
def prim_ACCEPT(inner, cur, ip):
    """GForth core 2012: receive a string of at most +n1 characters from input."""
    max_count = inner.pop_ds_int()
    addr = inner.pop_ds_int()

    stdin = inner.stdin
    line = stdin.readline()
    # Remove trailing newline if present
    line_len = len(line)
    if line_len > 0 and line[line_len - 1] == '\n':
        new_len = line_len - 1
        assert new_len >= 0  # Help RPython prove non-negative
        line = line[:new_len]

    # Limit to max_count
    line_len = len(line)
    if line_len > max_count and max_count >= 0:
        line = line[:max_count]

    # Store characters at addr
    final_len = len(line)
    for j in range(final_len):
        inner.cell_store(addr+j, ord(line[j]))

    inner.push_ds_int(final_len)
    return ip

# U.R ( u n -- )
@dont_look_inside
def prim_UDOTR(inner, cur, ip):
    """Display unsigned number right-justified in n-character field."""
    n = inner.pop_ds_int()
    u = inner.pop_ds_int()

    # Get unsigned value
    if u < 0:
        # Convert to unsigned (handle as positive for display)
        BIT_MASK = (1 << LONG_BIT) - 1  # 64-bit mask
        u = u & BIT_MASK

    num_str = str(u)
    width = n

    stdout = inner.stdout
    # Right-justify: add leading spaces if needed
    if len(num_str) < width:
        stdout.write(' ' * (width - len(num_str)))
    stdout.write(num_str)
    stdout.flush()
    return ip

# CodeThread-aware primitives

# LIT ( -- x )
def prim_LIT(inner, cur, ip):
    """GForth core 2012: push the next compilation literal."""
    lit = promote(cur.lits[ip - 1])
    if isinstance(lit, W_IntObject):
        inner.push_ds_int(lit.intval)
    elif isinstance(lit, W_FloatObject):
        inner.push_ds_float(lit.floatval)
    else:
        inner.push_ds(lit)
    return ip


# EXIT ( -- )
def prim_EXIT(inner, cur, ip):
    """GForth core 2012: terminate the current definition."""
    from rpyforth.inner_interp import EXIT_SENTINEL
    return EXIT_SENTINEL


# (ABORT") ( flag c-addr u -- )
def prim_ABORT_QUOTE_RUNTIME(inner, cur, ip):
    """Runtime for ABORT" - abort if flag is non-zero, printing message."""
    u = inner.pop_ds_int()
    c_addr = inner.pop_ds()
    flag = inner.pop_ds_int()

    if flag != 0:
        # Print the abort message
        if isinstance(c_addr, W_StringObject):
            msg = c_addr.strval
        else:
            msg = "ABORT"
        stdout = inner.stdout
        stdout.write("ABORT: ")
        stdout.write(msg)
        stdout.write("\n")
        # Clear stacks
        inner.ds_ptr_ints = 0
        inner.ds_ptr_floats = 0
        inner.ds_ptr_locals = 0
        inner.rs_ptr = 0
        inner.cs_ptr = 0  # Also clear call stack
        # Signal abort by returning EXIT_SENTINEL
        from rpyforth.inner_interp import EXIT_SENTINEL
        return EXIT_SENTINEL
    return ip


# Floating point operations

# F* ( f1 f2 -- f3 )
def prim_FMUL(inner, cur, ip):
    """Multiply two floating point numbers."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    #assert isinstance(f1, W_FloatObject)
    #assert isinstance(f2, W_FloatObject)
    inner.push_ds_float(f1 * f2)
    return ip


# F+ ( f1 f2 -- f3 )
def prim_FADD(inner, cur, ip):
    """Add two floating point numbers."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f1 + f2)
    return ip


# F- ( f1 f2 -- f3 )
def prim_FSUB(inner, cur, ip):
    """Subtract f2 from f1."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f1 - f2)
    return ip


# F/ ( f1 f2 -- f3 )
def prim_FDIV(inner, cur, ip):
    """Divide f1 by f2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f1 / f2)
    return ip


# F> ( f1 f2 -- flag )
def prim_FGREATER(inner, cur, ip):
    """Compare if f1 > f2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    if f1 > f2:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# FSWAP ( f1 f2 -- f2 f1 )
def prim_FSWAP(inner, cur, ip):
    """Exchange the top two floating point stack items."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f2)
    inner.push_ds_float(f1)
    return ip


# Return Stack Operations

# >R ( x -- ) ( R: -- x )
def prim_TORETURN(inner, cur, ip):
    """GForth core 2012: move x from data stack to return stack."""
    ds_ptr = inner.ds_ptr_ints
    x = inner.ds_ints[ds_ptr - 1]
    inner.ds_ptr_ints = ds_ptr - 1
    rs_ptr = inner.rs_ptr
    inner.rs[rs_ptr] = x
    inner.rs_ptr = rs_ptr + 1
    return ip


# R> ( -- x ) ( R: x -- )
def prim_FROMRETURN(inner, cur, ip):
    """GForth core 2012: move x from return stack to data stack."""
    rs_ptr = inner.rs_ptr - 1
    x = inner.rs[rs_ptr]
    inner.rs_ptr = rs_ptr
    inner.push_ds_int(x)
    return ip


# R@ ( -- x ) ( R: x -- x )
def prim_RFETCH(inner, cur, ip):
    """GForth core 2012: copy x from top of return stack to data stack."""
    x = inner.rs[inner.rs_ptr - 1]
    inner.push_ds_int(x)
    return ip


# 2>R ( x1 x2 -- ) ( R: -- x1 x2 )
def prim_2TORETURN(inner, cur, ip):
    """GForth core 2012: move x1 and x2 from data stack to return stack."""
    ds_ptr = inner.ds_ptr_ints
    x2 = inner.ds_ints[ds_ptr - 1]
    x1 = inner.ds_ints[ds_ptr - 2]
    inner.ds_ptr_ints = ds_ptr - 2
    rs_ptr = inner.rs_ptr
    inner.rs[rs_ptr] = x1
    inner.rs[rs_ptr + 1] = x2
    inner.rs_ptr = rs_ptr + 2
    return ip


# 2R> ( -- x1 x2 ) ( R: x1 x2 -- )
def prim_2FROMRETURN(inner, cur, ip):
    """GForth core 2012: move x1 and x2 from return stack to data stack."""
    rs_ptr = inner.rs_ptr
    x2 = inner.rs[rs_ptr - 1]
    x1 = inner.rs[rs_ptr - 2]
    inner.rs_ptr = rs_ptr - 2
    inner.push_ds_int(x1)
    inner.push_ds_int(x2)
    return ip


# 2R@ ( -- x1 x2 ) ( R: x1 x2 -- x1 x2 )
def prim_2RFETCH(inner, cur, ip):
    """GForth core 2012: copy x1 and x2 from top of return stack to data stack."""
    rs_ptr = inner.rs_ptr
    x2 = inner.rs[rs_ptr - 1]
    x1 = inner.rs[rs_ptr - 2]
    inner.push_ds_int(x1)
    inner.push_ds_int(x2)
    return ip

# PICK ( xu ... x1 x0 u -- xu ... x1 x0 xu )
def prim_PICK(inner, cur, ip):
    """Copy the u-th stack item to the top (0 PICK is equivalent to DUP)."""
    u = inner.pop_ds_int()

    # With virtualization disabled we can index directly for O(1) PICK
    target_index = inner.ds_ptr_ints - 1 - u
    assert target_index >= 0
    inner.push_ds_int(inner.ds_ints[target_index])
    return ip


# Floating point conversion and storage

# S>F ( n -- ) ( F: -- f )
def prim_S2F(inner, cur, ip):
    """Convert signed integer to float."""
    n = inner.pop_ds_int()
    inner.push_ds_float(float(n))
    return ip


# F! ( f-addr -- ) ( F: f -- )
def prim_FSTORE(inner, cur, ip):
    """Store float at address."""
    addr = inner.pop_ds_int()
    val = inner.pop_ds_float()
    inner.float_store(addr, val)
    return ip


# F@ ( f-addr -- ) ( F: -- f )
def prim_FFETCH(inner, cur, ip):
    """Fetch float from address."""
    addr = inner.pop_ds_int()
    w_float = inner.float_fetch(addr)
    assert isinstance(w_float, W_FloatObject)
    inner.push_ds_float(w_float.floatval)
    return ip


# FDUP ( F: f -- f f )
def prim_FDUP(inner, cur, ip):
    """Duplicate float on stack."""
    f = inner.pop_ds_float()
    inner.push_ds_float(f)
    inner.push_ds_float(f)
    return ip


# Dictionary Operations

# EXECUTE ( xt -- )
def prim_EXECUTE(inner, cur, ip):
    """GForth core 2012: execute the execution token xt."""
    xt = inner.pop_ds()
    assert isinstance(xt, W_WordObject)
    word = xt.word
    inner.execute_word_now(word)
    return ip


# >BODY ( xt -- a-addr )
def prim_TOBODY(inner, cur, ip):
    """GForth core 2012: return the parameter field address corresponding to xt."""
    xt = inner.pop_ds()
    assert isinstance(xt, W_WordObject)
    word = xt.word
    # For words created with CREATE, VARIABLE, CONSTANT, etc.,
    # the body is in the first literal of the code thread
    if word.thread is not None and len(word.thread.lits) > 0:
        body = word.thread.lits[0]
        if isinstance(body, W_IntObject):
            inner.push_ds_int(body.intval)
        else:
            inner.push_ds(body)
    else:
        # For primitive words, there's no body
        # Push 0 or raise an error
        inner.push_ds_int(0)
    return ip


# System Operations

# FILL ( c-addr u char -- )
def prim_FILL(inner, cur, ip):
    """GForth core 2012: fill u bytes of memory starting at c-addr with char."""
    char = inner.pop_ds_int()
    u = inner.pop_ds_int()
    addr = inner.pop_ds_int()

    # Fill memory with char
    for i in range(u):
        inner.cell_store(addr+i, char)
    return ip


# MOVE ( addr1 addr2 u -- )
def prim_MOVE(inner, cur, ip):
    """GForth core 2012: copy u bytes from addr1 to addr2."""
    u = inner.pop_ds_int()
    addr2 = inner.pop_ds_int()
    addr1 = inner.pop_ds_int()

    # Copy u bytes from addr1 to addr2
    # Handle overlapping regions by using a temporary buffer
    src = addr1
    dst = addr2

    # Read all values first (in case of overlap)
    values = []
    for i in range(u):
        w_x = inner.cell_fetch(src+i)
        assert isinstance(w_x, W_IntObject)
        values.append(w_x.intval)

    # Write to destination
    for i in range(u):
        inner.cell_store(dst+i, values[i])
    return ip


# Memory Access Operations (additional)

# +! ( n|u a-addr -- )
def prim_PLUSSTORE(inner, cur, ip):
    """GForth core 2012: add n to the value stored at a-addr."""
    addr = inner.pop_ds_int()
    n = inner.pop_ds_int()
    # Fetch current value, add n, store back
    current = inner.cell_fetch(addr)
    assert isinstance(current, W_IntObject)
    new_val = current.intval + n
    inner.cell_store(addr, new_val)
    return ip


# 2@ ( a-addr -- x1 x2 )
def prim_2FETCH(inner, cur, ip):
    """GForth core 2012: fetch the cell pair stored at a-addr."""
    addr = inner.pop_ds_int()
    # Actual storage from cell_2store: x1 at addr, x2 at addr+cell
    x1 = inner.cell_fetch(addr)
    addr2 = addr + inner.cell_size_bytes
    x2 = inner.cell_fetch(addr2)
    # Push x1 first, then x2 to get stack ( x1 x2 )
    assert isinstance(x1, W_IntObject)
    assert isinstance(x2, W_IntObject)
    inner.push_ds_int(x1.intval)
    inner.push_ds_int(x2.intval)
    return ip


# C! ( char c-addr -- )
def prim_C_STORE(inner, cur, ip):
    """GForth core 2012: store char at c-addr."""
    addr = inner.pop_ds_int()
    char = inner.pop_ds_int()
    # Store just the character (we'll use cell_store for simplicity)
    inner.cell_store(addr, char)
    return ip


# C@ ( c-addr -- char )
def prim_C_FETCH(inner, cur, ip):
    """GForth core 2012: fetch the character stored at c-addr."""
    addr = inner.pop_ds_int()
    char = inner.cell_fetch(addr)
    assert isinstance(char, W_IntObject)
    inner.push_ds_int(char.intval)
    return ip


# CHAR+ ( c-addr1 -- c-addr2 )
def prim_CHAR_PLUS(inner, cur, ip):
    """GForth core 2012: add the size of a character to c-addr1."""
    addr = inner.pop_ds_int()
    # In our implementation, characters are 1 byte
    inner.push_ds_int(addr + 1)
    return ip


# CHARS ( n1 -- n2 )
def prim_CHARS(inner, cur, ip):
    """GForth core 2012: convert n1 characters to address units."""
    n = inner.pop_ds_int()
    # In our implementation, 1 char = 1 address unit
    inner.push_ds_int(n)
    return ip


# ALIGN ( -- )
def prim_ALIGN(inner, cur, ip):
    """GForth core 2012: align the data-space pointer."""
    # Align to cell boundary
    remainder = inner.here % inner.cell_size_bytes
    if remainder != 0:
        inner.here += (inner.cell_size_bytes - remainder)
    return ip


# ALIGNED ( addr -- a-addr )
def prim_ALIGNED(inner, cur, ip):
    """GForth core 2012: return the aligned address."""
    addr_val = inner.pop_ds_int()
    remainder = addr_val % inner.cell_size_bytes
    if remainder != 0:
        addr_val += (inner.cell_size_bytes - remainder)
    inner.push_ds_int(addr_val)
    return ip


# Data Space Operations

# HERE ( -- addr )
def prim_HERE(inner, cur, ip):
    """GForth core 2012: return the address of the next available data space location."""
    inner.push_ds_int(inner.here)
    return ip


# , ( x -- )
def prim_COMMA(inner, cur, ip):
    """GForth core 2012: reserve one cell of data space and store x in it."""
    x = inner.pop_ds_int()
    addr = inner.here
    inner.cell_store(addr, x)
    inner.here += inner.cell_size_bytes
    return ip


# C, ( char -- )
def prim_C_COMMA(inner, cur, ip):
    """GForth core 2012: reserve one character of data space and store char in it."""
    char = inner.pop_ds_int()
    # For simplicity, we'll use cell_store but only increment by 1 byte
    addr = inner.here
    inner.cell_store(addr, char)
    inner.here += 1
    return ip


# ALLOT ( n -- )
def prim_ALLOT(inner, cur, ip):
    """GForth core 2012: reserve n address units of data space."""
    n = inner.pop_ds_int()
    inner.here += n
    return ip


# Comparison

# = ( x1 x2 -- flag )
def prim_EQUAL(inner, cur, ip):
    """GForth core 2012: flag is true when x1 equals x2."""
    ptr = inner.ds_ptr_ints
    x2 = inner.ds_ints[ptr - 1]
    x1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = -1 if x1 == x2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip


# System Operations

# BYE ( -- )
def prim_BYE(inner, cur, ip):
    """GForth core 2012: exit the Forth system."""
    from rpyforth.inner_interp import Bye
    raise Bye()


# <= ( n1 n2 -- flag )
def prim_LESSEQUAL(inner, cur, ip):
    """Flag is true when n1 is less than or equal to n2."""
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    if n1 <= n2:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# >= ( n1 n2 -- flag )
def prim_GREATEREQUAL(inner, cur, ip):
    """Flag is true when n1 is greater than or equal to n2."""
    n2 = inner.pop_ds_int()
    n1 = inner.pop_ds_int()
    if n1 >= n2:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# <> ( n1 n2 -- flag )
def prim_NOTEQUAL(inner, cur, ip):
    """Flag is true when n1 is not equal to n2."""
    ptr = inner.ds_ptr_ints
    n2 = inner.ds_ints[ptr - 1]
    n1 = inner.ds_ints[ptr - 2]
    inner.ds_ints[ptr - 2] = -1 if n1 != n2 else 0
    inner.ds_ptr_ints = ptr - 1
    return ip


# F< ( r1 r2 -- flag )
def prim_FLESS(inner, cur, ip):
    """Flag is true when r1 is less than r2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    if f1 < f2:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# F= ( r1 r2 -- flag )
def prim_FEQUAL(inner, cur, ip):
    """Flag is true when r1 equals r2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    if f1 == f2:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# F0< ( r -- flag )
def prim_FZEROLESS(inner, cur, ip):
    """Flag is true when r is less than zero."""
    f = inner.pop_ds_float()
    if f < 0.0:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# F0= ( r -- flag )
def prim_FZEROEQUAL(inner, cur, ip):
    """Flag is true when r equals zero."""
    f = inner.pop_ds_float()
    if f == 0.0:
        inner.push_ds_int(-1)
    else:
        inner.push_ds_int(0)
    return ip


# FNEGATE ( r1 -- r2 )
def prim_FNEGATE(inner, cur, ip):
    """Negate r1."""
    f = inner.pop_ds_float()
    inner.push_ds_float(-f)
    return ip


# FABS ( r1 -- r2 )
def prim_FABS(inner, cur, ip):
    """Return absolute value of r1."""
    f = inner.pop_ds_float()
    if f < 0.0:
        inner.push_ds_float(-f)
    else:
        inner.push_ds_float(f)
    return ip


# FOVER ( r1 r2 -- r1 r2 r1 )
def prim_FOVER(inner, cur, ip):
    """Copy second float to top of float stack."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f1)
    inner.push_ds_float(f2)
    inner.push_ds_float(f1)
    return ip


# FDROP ( r -- )
def prim_FDROP(inner, cur, ip):
    """Drop top float from stack."""
    inner.pop_ds_float()
    return ip


# FROT ( r1 r2 r3 -- r2 r3 r1 )
def prim_FROT(inner, cur, ip):
    """Rotate top three floats."""
    f3 = inner.pop_ds_float()
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    inner.push_ds_float(f2)
    inner.push_ds_float(f3)
    inner.push_ds_float(f1)
    return ip


# FLOAT+ ( f-addr1 -- f-addr2 )
def prim_FLOATPLUS(inner, cur, ip):
    """Add size of float to address."""
    addr = inner.pop_ds_int()
    inner.push_ds_int(addr + 8)  # 8 bytes for a double
    return ip


# ALLOCATE ( u -- a-addr ior )
def prim_ALLOCATE(inner, cur, ip):
    """Allocate u bytes of memory, return address and 0 (success) or non-zero (failure)."""
    size = inner.pop_ds_int()
    # Use the inner interpreter's memory buffer
    # Allocate from 'here' and advance
    addr = inner.here
    inner.here = addr + size
    # Initialize memory (ensure we have space in mem array)
    if inner.here < len(inner.mem):
        inner.push_ds_int(addr)
        inner.push_ds_int(0)  # success
    else:
        inner.push_ds_int(0)
        inner.push_ds_int(-1)  # failure
    return ip


# FREE ( a-addr -- ior )
def prim_FREE(inner, cur, ip):
    """Free previously allocated memory. Always succeeds in this simple implementation."""
    inner.pop_ds_int()  # Discard address
    inner.push_ds_int(0)  # Always success (no actual deallocation)
    return ip


# THROW ( k*x n -- k*x | i*x n )
def prim_THROW(inner, cur, ip):
    """Throw exception. If n is 0, do nothing. Otherwise, abort with message."""
    n = inner.pop_ds_int()
    if n != 0:
        print "THROW: exception", n
        import os
        os._exit(1)
    return ip


# Precision for floating point output
_float_precision = [6]  # Default precision, stored as list for mutability


# SET-PRECISION ( u -- )
def prim_SET_PRECISION(inner, cur, ip):
    """Set number of significant digits for floating point output."""
    prec = inner.pop_ds_int()
    _float_precision[0] = prec
    return ip


# PRECISION ( -- u )
def prim_PRECISION(inner, cur, ip):
    """Return current floating point precision."""
    inner.push_ds_int(_float_precision[0])
    return ip


# F. ( r -- )
def prim_FDOT(inner, cur, ip):
    """Print floating point number."""
    f = inner.pop_ds_float()
    prec = _float_precision[0]
    result = formatd(f, 'g', prec)
    stdout = inner.stdout
    stdout.write(result + " ")
    stdout.flush()
    return ip


# D>F ( d -- r )
def prim_D2F(inner, cur, ip):
    """Convert double-cell integer to float."""
    high = inner.pop_ds_int()
    low = inner.pop_ds_int()
    if LONG_BIT == 64:
        d = low  # or combine if needed
    else:
        d = (high << 32) | (low & 0xFFFFFFFF)
    inner.push_ds_float(float(d))
    return ip


# F>D ( r -- d )
def prim_F2D(inner, cur, ip):
    """Convert float to double-cell integer (truncate toward zero)."""
    f = inner.pop_ds_float()
    d = int(f)
    # Push as double (two cells)
    if LONG_BIT == 64:
        inner.push_ds_int(d)
        inner.push_ds_int(0)  # high part
    else:
        inner.push_ds_int(d & 0xFFFFFFFF)  # low
        inner.push_ds_int(d >> 32)  # high
    return ip


# FMIN ( r1 r2 -- r3 )
def prim_FMIN(inner, cur, ip):
    """Return the lesser of r1 and r2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    if f1 < f2:
        inner.push_ds_float(f1)
    else:
        inner.push_ds_float(f2)
    return ip


# FMAX ( r1 r2 -- r3 )
def prim_FMAX(inner, cur, ip):
    """Return the greater of r1 and r2."""
    f2 = inner.pop_ds_float()
    f1 = inner.pop_ds_float()
    if f1 > f2:
        inner.push_ds_float(f1)
    else:
        inner.push_ds_float(f2)
    return ip


# FLOOR ( r1 -- r2 )
def prim_FLOOR(inner, cur, ip):
    """Round r1 toward negative infinity."""
    import math
    f = inner.pop_ds_float()
    inner.push_ds_float(math.floor(f))
    return ip


# FROUND ( r1 -- r2 )
def prim_FROUND(inner, cur, ip):
    """Round r1 to nearest integer."""
    f = inner.pop_ds_float()
    inner.push_ds_float(float(int(f + 0.5 if f >= 0 else f - 0.5)))
    return ip


# FLITERAL ( F: r -- ) compilation; ( -- F: r ) run-time
def prim_FLITERAL(inner, cur, ip):
    """Compile a float literal. At run-time, push the float onto the float stack."""
    # Pop float from float stack
    if inner.ds_ptr_floats > 0:
        floatval = inner.pop_ds_float()
        # Emit LIT <float> into the compilation buffer
        outer = inner.outer
        if outer is not None:
            outer._emit_lit(W_FloatObject(floatval))
    elif inner.ds_ptr_locals > 0:
        val = inner.pop_ds()
        outer = inner.outer
        if outer is not None:
            outer._emit_lit(val)
    else:
        print "FLITERAL: float stack underflow"
    return ip


# Time-related primitives

# MS ( u -- )
def prim_MS(inner, cur, ip):
    """Wait at least u milliseconds."""
    import time
    u = inner.pop_ds_int()
    if u > 0:
        # Convert milliseconds to seconds for time.sleep
        time.sleep(u / 1000.0)
    return ip


# TIME&DATE ( -- nsec nmin nhour nday nmonth nyear )
def prim_TIME_AND_DATE(inner, cur, ip):
    raise NotImplementedError


# UTIME ( -- d )
def prim_UTIME(inner, cur, ip):
    from rpython.rlib.rtime import time
    # Get current time in seconds (float), convert to microseconds
    t = time()
    usecs = int(t * 1000000.0)
    # Push as double-cell (low, high)
    BIT_MASK = (1 << LONG_BIT) - 1
    low = usecs & BIT_MASK
    high = usecs >> LONG_BIT
    inner.push_ds_int(low)
    inner.push_ds_int(high)
    return ip


# CPUTIME ( -- duser dsystem )
def prim_CPUTIME(inner, cur, ip):
    """Report CPU times in microseconds. duser is user-level CPU time, dsystem is system-level CPU time."""
    # Try to get process times if available
    try:
        import os
        # os.times() returns (user, system, children_user, children_system, elapsed)
        times = os.times()
        user_secs = times[0]
        sys_secs = times[1]
    except:
        # Fallback: use elapsed time for user, 0 for system
        from rpython.rlib.rtime import time
        user_secs = time()
        sys_secs = 0.0

    # Convert to microseconds
    user_usecs = int(user_secs * 1000000.0)
    sys_usecs = int(sys_secs * 1000000.0)

    BIT_MASK = (1 << LONG_BIT) - 1

    # Push duser (low, high)
    user_low = user_usecs & BIT_MASK
    user_high = user_usecs >> LONG_BIT
    inner.push_ds_int(user_low)
    inner.push_ds_int(user_high)

    # Push dsystem (low, high)
    sys_low = sys_usecs & BIT_MASK
    sys_high = sys_usecs >> LONG_BIT
    inner.push_ds_int(sys_low)
    inner.push_ds_int(sys_high)
    return ip


# LITERAL ( x -- ) compilation; ( -- x ) run-time
def prim_LITERAL(inner, cur, ip):
    """Compile a literal. At run-time, push the value onto the stack."""
    # Pop value from stack
    if inner.ds_ptr_ints > 0:
        intval = inner.pop_ds_int()
        outer = inner.outer
        if outer is not None:
            outer._emit_lit(W_IntObject(intval))
    elif inner.ds_ptr_floats > 0:
        floatval = inner.pop_ds_float()
        outer = inner.outer
        if outer is not None:
            outer._emit_lit(W_FloatObject(floatval))
    elif inner.ds_ptr_locals > 0:
        val = inner.pop_ds()
        outer = inner.outer
        if outer is not None:
            outer._emit_lit(val)
    else:
        print "LITERAL: stack underflow"
    return ip


def install_primitives(outer):
    outer.define_prim("0=", prim_ZEROEQUAL)
    outer.define_prim("0<", prim_ZEROLESS)
    outer.define_prim("0>", prim_ZEROGREATER)
    outer.define_prim(">",  prim_GREATER)
    outer.define_prim("<",  prim_LESS)
    outer.define_prim("0<>", prim_ZERONOTEQUAL)
    outer.define_prim("U<", prim_U_LESS)
    # stack manipulation
    outer.define_prim("DUP", prim_DUP)
    outer.define_prim("DROP", prim_DROP)
    outer.define_prim("NIP", prim_NIP)
    outer.define_prim("SWAP", prim_SWAP)
    outer.define_prim("OVER", prim_OVER)

    outer.define_prim("2DUP", prim_2DUP)
    outer.define_prim("2DROP", prim_2DROP)
    outer.define_prim("2SWAP", prim_2SWAP)
    outer.define_prim("2OVER", prim_2OVER)

    outer.define_prim("?DUP", prim_QUESTIONDUP)

    outer.define_prim("ROT", prim_ROT)
    outer.define_prim("-ROT", prim_NROT)
    outer.define_prim("MAX", prim_MAX)
    outer.define_prim("MIN", prim_MIN)

    outer.define_prim("DEPTH", prim_DEPTH)

    outer.define_prim("RSHIFT", prim_RSHIFT)
    outer.define_prim("LSHIFT", prim_LSHIFT)

    outer.define_prim("S>D", prim_S_TO_D)
    outer.define_prim("D+", prim_D_PLUS)
    outer.define_prim("D-", prim_D_MINUS)
    outer.define_prim("D.", prim_D_DOT)
    outer.define_prim("BL", prim_BL)
    outer.define_prim("FILL", prim_FILL)
    outer.define_prim("MOVE", prim_MOVE)

    outer.define_prim("2*", prim_2STAR)
    outer.define_prim("2/", prim_2SLASH)

    # arithmetic
    outer.define_prim("+", prim_ADD)
    outer.define_prim("-", prim_SUB)
    outer.define_prim("*", prim_MUL)
    outer.define_prim("/", prim_DIV)
    outer.define_prim("*/", prim_MUL_SLASH)
    outer.define_prim("/MOD", prim_DIV_MOD)
    outer.define_prim("*/MOD", prim_MUL_DIV_MOD)

    outer.define_prim("ABS", prim_ABS)
    outer.define_prim("NEGATE", prim_NEGATE)
    outer.define_prim("MOD", prim_MOD)
    outer.define_prim("/", prim_DIV)
    outer.define_prim("/MOD", prim_DIVMOD)
    outer.define_prim("*/", prim_STARSLASH)
    outer.define_prim("*/MOD", prim_STARSLASHMOD)
    outer.define_prim("FM/MOD", prim_FMSLASHMOD)
    outer.define_prim("SM/REM", prim_SMSLASHREM)
    outer.define_prim("UM/MOD", prim_UMSLASHMOD)

    outer.define_prim("1+", prim_INC)
    outer.define_prim("1-", prim_DEC)

    outer.define_prim("M*", prim_MUL_STAR)
    outer.define_prim("UM*", prim_U_MUL_STAR)

    outer.define_prim("AND", prim_AND)
    outer.define_prim("OR", prim_OR)
    outer.define_prim("XOR", prim_XOR)
    outer.define_prim("INVERT", prim_INVERT)

    # comparison
    outer.define_prim("U<", prim_ULESS)

    outer.define_prim("FM/MOD", prim_FM_DIV_MOD)
    outer.define_prim("UM/MOD", prim_UM_DIV_MOD)
    outer.define_prim("SM/REM", prim_SM_DIV_REM)

    # I/O
    outer.define_prim(".", prim_DOT)
    outer.define_prim("U.", prim_U_DOT)
    outer.define_prim("EMIT", prim_EMIT)
    outer.define_prim("SPACE", prim_SPACE)
    outer.define_prim("SPACES", prim_SPACES)
    outer.define_prim("CR", prim_CR)
    outer.define_prim("U.", prim_UDOT)
    outer.define_prim("KEY", prim_KEY)
    outer.define_prim("ACCEPT", prim_ACCEPT)
    outer.define_prim("U.R", prim_UDOTR)

    # memory management
    outer.define_prim("!", prim_STORE)
    outer.define_prim("2!", prim_2STORE)
    outer.define_prim("@", prim_FETCH)
    outer.define_prim("CELL", prim_CELL)
    outer.define_prim("CELL+", prim_CELLPLUS)
    outer.define_prim("CELLS", prim_CELLS)
    outer.define_prim("+!", prim_PLUSSTORE)
    outer.define_prim("2@", prim_2FETCH)
    outer.define_prim("C!", prim_C_STORE)
    outer.define_prim("C@", prim_C_FETCH)
    outer.define_prim("CHAR+", prim_CHAR_PLUS)
    outer.define_prim("CHARS", prim_CHARS)
    outer.define_prim("ALIGN", prim_ALIGN)
    outer.define_prim("ALIGNED", prim_ALIGNED)

    # BASE
    outer.define_prim("BASE@", prim_BASE_FETCH)
    outer.define_prim("BASE!", prim_BASE_STORE)
    outer.define_prim("DECIMAL", prim_DECIMAL)
    outer.define_prim("HEX", prim_HEX)
    outer.define_prim("OCTAL", prim_OCTAL)
    outer.define_prim("BINARY", prim_BINARY)

    outer.define_prim("<#", prim_LESSNUM)
    outer.define_prim("#", prim_NUMSIGN)
    outer.define_prim("#S", prim_NUMSIGN_S)
    outer.define_prim("#>", prim_NUMGREATER)
    outer.define_prim("HOLD", prim_HOLD)
    outer.define_prim("SIGN", prim_SIGN)

    outer.define_prim("TYPE", prim_TYPE)  # for testing output

    # loop
    outer.define_prim("0BRANCH", prim_0BRANCH)
    outer.define_prim("BRANCH", prim_BRANCH)
    outer.define_prim("(DO)", prim_DO_RUNTIME)
    outer.define_prim("(LOOP)", prim_LOOP_RUNTIME)
    outer.define_prim("(+LOOP)", prim_PLUSLOOP_RUNTIME)
    outer.define_prim("UNLOOP", prim_UNLOOP)
    outer.define_prim("LEAVE", prim_LEAVE)
    outer.define_prim("I", prim_I)
    outer.define_prim("J", prim_J)

    # thread ops
    outer.define_prim("LIT", prim_LIT)
    outer.define_prim("EXIT", prim_EXIT)
    outer.define_prim("(ABORT\")", prim_ABORT_QUOTE_RUNTIME)

    # floating point
    outer.define_prim("F*", prim_FMUL)
    outer.define_prim("F+", prim_FADD)
    outer.define_prim("F-", prim_FSUB)
    outer.define_prim("F/", prim_FDIV)
    outer.define_prim("F>", prim_FGREATER)
    outer.define_prim("F<", prim_FLESS)
    outer.define_prim("F=", prim_FEQUAL)
    outer.define_prim("F0<", prim_FZEROLESS)
    outer.define_prim("F0=", prim_FZEROEQUAL)
    outer.define_prim("FSWAP", prim_FSWAP)
    outer.define_prim("S>F", prim_S2F)
    outer.define_prim("F!", prim_FSTORE)
    outer.define_prim("F@", prim_FFETCH)
    outer.define_prim("FDUP", prim_FDUP)
    outer.define_prim("FDROP", prim_FDROP)
    outer.define_prim("FOVER", prim_FOVER)
    outer.define_prim("FROT", prim_FROT)
    outer.define_prim("FNEGATE", prim_FNEGATE)
    outer.define_prim("FABS", prim_FABS)
    outer.define_prim("FMIN", prim_FMIN)
    outer.define_prim("FMAX", prim_FMAX)
    outer.define_prim("FLOOR", prim_FLOOR)
    outer.define_prim("FROUND", prim_FROUND)
    outer.define_prim("FLOAT", prim_FLOAT)
    outer.define_prim("FLOATS", prim_FLOATS)
    outer.define_prim("FLOAT+", prim_FLOATPLUS)
    outer.define_prim("F.", prim_FDOT)
    outer.define_prim("D>F", prim_D2F)
    outer.define_prim("F>D", prim_F2D)
    outer.define_prim("SET-PRECISION", prim_SET_PRECISION)
    outer.define_prim("PRECISION", prim_PRECISION)

    # stack manipulation
    outer.define_prim("PICK", prim_PICK)

    # return stack
    outer.define_prim(">R", prim_TORETURN)
    outer.define_prim("R>", prim_FROMRETURN)
    outer.define_prim("R@", prim_RFETCH)
    outer.define_prim("2>R", prim_2TORETURN)
    outer.define_prim("2R>", prim_2FROMRETURN)
    outer.define_prim("2R@", prim_2RFETCH)

    # dictionary
    outer.define_prim("EXECUTE", prim_EXECUTE)
    outer.define_prim(">BODY", prim_TOBODY)

    # data space
    outer.define_prim("HERE", prim_HERE)
    outer.define_prim(",", prim_COMMA)
    outer.define_prim("C,", prim_C_COMMA)
    outer.define_prim("ALLOT", prim_ALLOT)

    # comparison
    outer.define_prim("=", prim_EQUAL)
    outer.define_prim("<=", prim_LESSEQUAL)
    outer.define_prim(">=", prim_GREATEREQUAL)
    outer.define_prim("<>", prim_NOTEQUAL)

    # memory allocation
    outer.define_prim("ALLOCATE", prim_ALLOCATE)
    outer.define_prim("FREE", prim_FREE)

    # exception handling
    outer.define_prim("THROW", prim_THROW)

    # system
    outer.define_prim("BYE", prim_BYE)

    # time
    outer.define_prim("MS", prim_MS)
    outer.define_prim("TIME&DATE", prim_TIME_AND_DATE)
    outer.define_prim("UTIME", prim_UTIME)
    outer.define_prim("CPUTIME", prim_CPUTIME)
