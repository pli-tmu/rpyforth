40 constant NUM

\ compute fibonacci numbers
: fib  recursive
    dup 2 <
    if
    drop 1
    else
    dup
        2 - fib
    swap
    1 - fib
    +
    then ;

NUM fib 1 u.r cr

bye
