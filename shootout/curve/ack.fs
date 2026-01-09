\ Ackermann function benchmark for JIT analysis (Corrected)

7 constant NUM
30 constant ITERATIONS

variable start-h variable start-l
variable end-h   variable end-l

: get-time ( -- d ) utime ;
: diff-time ( d-start d-end -- d-diff ) 2swap d- ;

\ ---------------------------------------------------------
\ Target Workload: Ackermann
\ Stack effect: ( n m -- res )  <-- 注意: m がスタックトップ
\ ---------------------------------------------------------
: ack ( n m -- res )
    recursive
    dup 0= if
        drop 1+
    else
        swap dup 0= if
            drop 1- 1 swap ack
        else
            1- over 1- rot rot swap ack swap ack
        then
    then ;

\ ---------------------------------------------------------
\ Benchmark Driver
\ ---------------------------------------------------------
: run-benchmark
    ." Iteration,Time(usec)" cr

    ITERATIONS 0 do
        get-time

        NUM 3 ack drop

        get-time
        diff-time
        i . ." ," d. cr
    loop
;

run-benchmark
bye
