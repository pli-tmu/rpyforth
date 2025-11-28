30 constant NUM

: nestedloops
  NUM 0 do
    NUM 0 do
      NUM 0 do
        NUM 0 do
          NUM 0 do
            NUM 0 do
              1+
            loop
          loop
        loop
      loop
    loop
  loop ;

\ run test and print result
: main
    1 0 DO
        0 nestedloops 1
    LOOP
;

main
