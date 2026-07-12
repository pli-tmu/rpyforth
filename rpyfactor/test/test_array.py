from rpyfactor.test.conftest import run, run_result_int
from rpyfactor.values import W_Array


def test_new_array_zeros():
    interp = run("5 <array>")
    arr = interp.st().pop()
    assert isinstance(arr, W_Array)
    assert [x.val for x in arr.items] == [0, 0, 0, 0, 0]


def test_nth_on_array():
    assert run_result_int("5 <array> 2 swap nth") == 0


def test_set_nth_mutates():
    src = "5 <array> 7 2 rot set-nth"
    interp = run(src)
    # set-nth consumed the array; rebuild via a variant that keeps it.
    assert interp.st().size() == 0


def test_set_nth_then_nth():
    src = "5 <array> dup 7 2 rot set-nth 2 swap nth"
    assert run_result_int(src) == 7


def test_set_nth_shares_identity():
    # dup shares the same underlying array: mutating one reference is
    # visible via the other (Phase B mutability requirement for nsieve).
    src = "3 <array> dup 9 0 rot set-nth 0 swap nth"
    assert run_result_int(src) == 9


def test_nth_on_list_literal():
    assert run_result_int("{ 10 20 30 } 1 swap nth") == 20

def test_length_on_array():
    assert run_result_int("9 <array> length") == 9


def test_size_on_array():
    assert run_result_int("9 <array> size") == 9


def test_nsieve_mutable_array_primes():
    # Small-scale rehearsal of benchmark/factor/phase-b/nsieve.factor's
    # mutable-array sieve: count primes <= 30 using <array>/nth/set-nth.
    src = """
    : pack3 ( a b c -- lst ) { } cons cons cons ;
    : unpack3 ( lst -- a b c )
        dup first swap
        dup rest first swap
        rest rest first ;

    : flags@ ( flags i -- flags i elt ) 2dup swap nth ;
    : poke1 ( flags j -- flags j ) 2dup 1 swap rot set-nth ;

    : build-multiples ( i -- list )
        dup dup *
        { } swap rot
        pack3
        [ dup rest first 30 <= ]
        [
            unpack3
            2dup +
            [ [ swap cons ] dip ] dip
            swap
            pack3
        ]
        while
        unpack3 drop drop ;

    : mark-multiples ( flags i -- flags i )
        swap over build-multiples [ poke1 drop ] step swap ;

    : primes ( -- count )
        31 <array> 2 0 pack3
        [ dup rest first 30 <= ]
        [
            unpack3
            [ flags@ ] dip
            swap 0 =
            [ 1+ [ mark-multiples ] dip ]
            [ ]
            if
            [ 1+ ] dip
            pack3
        ]
        while
        unpack3 nip nip ;

    primes
    """
    # primes <= 30: 2 3 5 7 11 13 17 19 23 29 -> 10
    assert run_result_int(src) == 10
