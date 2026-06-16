import os
import sys

from rpyforth.inner_interp import InnerInterpreter, Bye
from rpyforth.outer_interp import OuterInterpreter

from rpython.rlib import jit
from rpython.rlib.streamio import open_file_as_stream

def entry_point(argv):
    for i in range(len(argv)):
        if argv[i] == "--jit":
            if len(argv) == i + 1:
                print("missing argument after --jit")
                return 2
            jitarg = argv[i + 1]
            del argv[i:i+2]
            jit.set_user_param(None, jitarg)
            break

    if len(argv) < 2:
        print("Usage: %s filename x" % (argv[0],))
        return 2

    jit.set_user_param(None, "trace_limit=200000")
    inner = InnerInterpreter()
    outer = OuterInterpreter(inner)
    path = argv[1]
    args = []
    if len(argv) > 2:
        args = argv[2:]
    inner.argv = args
    f = open_file_as_stream(path)
    try:
        for line in f.readall().split('\n'):
            outer.interpret_line(line)
    except Bye:
        pass
    f.close()
    return 0

def target(driver, args):
    driver.exe_name = os.environ.get("RPYFORTH_EXE_NAME", "rpyforth-%(backend)s")
    return entry_point, None

if __name__ == '__main__':
    import sys
    print(entry_point(sys.argv))
