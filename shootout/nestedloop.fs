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
0 nestedloops 1 U.R CR

bye
