"""Standard-library words defined in Forth and loaded at startup.

Words that are derivable from existing primitives live here rather than as
RPython primitives: each is a few lines of Forth, and straight-line ones are
inlined at their call sites. The text is embedded as a constant so the binary
carries no external file dependency."""

PRELUDE = """\
: TRUE -1 ;
: FALSE 0 ;
: TUCK SWAP OVER ;
: U> SWAP U< ;
: WITHIN OVER - >R - R> U< ;
: ERASE 0 FILL ;
: /STRING DUP >R - SWAP R> + SWAP ;
: BLANK BL FILL ;
: -TRAILING BEGIN dup WHILE 2dup + 1- c@ bl = IF 1- ELSE EXIT THEN REPEAT ;
: DNEGATE 0 0 2SWAP D- ;
: D0= OR 0= ;
: D2* 2DUP D+ ;
: CELL- 1 CELLS - ;
: M+ S>D D+ ;
: D= D- D0= ;
: D<> D= 0= ;
CREATE pad-buf 512 ALLOT
: PAD pad-buf ;
"""


def load_prelude(outer):
    for line in PRELUDE.split("\n"):
        outer.interpret_line(line)
