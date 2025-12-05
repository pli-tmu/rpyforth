\ Ackermann using continuation stack to achieve tail-call form
create ack-stack 100 cells allot
variable ack-sp

: ack-push ( n -- )
  ack-sp @ cells ack-stack + !
  1 ack-sp +! ;

: ack-pop ( -- n )
  -1 ack-sp +!
  ack-sp @ cells ack-stack + @ ;

: ack-tail ( m n -- result )
  over 0= if                    \ m=0: return n+1
    nip 1+
    ack-sp @ 0> if
      ack-pop recurse           \ continue with stacked work
    then
  else
    dup 0= if                   \ m>0, n=0: ack(m-1, 1)
      drop 1- 1 recurse
    else                        \ m>0, n>0: ack(m-1, ack(m, n-1))
      over ack-push             \ save m for later
      1- recurse                \ compute ack(m, n-1) first
      ack-pop 1- swap recurse   \ then ack(m-1, result)
    then
  then ;

: ack ( m n -- result )
  0 ack-sp !
  ack-tail ;
\ Test
3 3 ack .  \ Should print 125
