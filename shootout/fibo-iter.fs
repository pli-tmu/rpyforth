\ Tail-recursive Fibonacci using accumulators
: fibo-iter ( n acc1 acc2 -- result )
  rot dup 0= if
    drop drop        \ n is 0, return acc1
  else
    1- -rot          \ n-1, acc2, acc1
    over + swap      \ n-1, acc1+acc2, acc2
    recurse
  then ;

: fibo ( n -- result )
  0 1 fibo-iter ;

\ Test
: main
    10000 0 DO
        120 fibo DROP  \ Should print 55
    LOOP ;

main
