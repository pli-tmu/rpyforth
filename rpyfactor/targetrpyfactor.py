"""RPython translation entry point for rpyfactor."""

import os
import sys
import time

from rpython.rlib import jit
from rpython.rlib.streamio import open_file_as_stream

from rpyfactor.interp import Interpreter
from rpyfactor.values import FactorError, W_Int


def entry_point(argv):
    i = 0
    while i < len(argv):
        if argv[i] == "--jit":
            if i + 1 >= len(argv):
                print("missing argument after --jit")
                return 2
            jit.set_user_param(None, argv[i + 1])
            del argv[i:i + 2]
            continue
        i += 1

    if len(argv) < 2:
        print("Usage: %s [--jit PARAM] filename.factor" % (argv[0],))
        print("       %s -c 'program text'" % (argv[0],))
        return 2

    jit.set_user_param(None, "trace_limit=200000")

    source = ""
    if argv[1] == "-c":
        if len(argv) < 3:
            print("missing program after -c")
            return 2
        source = argv[2]
    else:
        f = open_file_as_stream(argv[1])
        try:
            source = f.readall()
        finally:
            f.close()

    interp = Interpreter()
    try:
        t0 = time.time()
        interp.run_source(source)
        elapsed_usec = int((time.time() - t0) * 1000000.0)
    except FactorError as exc:
        print("ERROR: %s" % exc.msg)
        return 1

    if interp.st().size() >= 1:
        v = interp.st().peek(0)
        if isinstance(v, W_Int):
            print("Result: %d" % v.val)
            print("Elapsed: %d usec" % elapsed_usec)
    return 0


def target(driver, args):
    driver.exe_name = os.environ.get("RPYFACTOR_EXE_NAME", "rpyfactor-%(backend)s")
    return entry_point, None


if __name__ == "__main__":
    sys.exit(entry_point(sys.argv))
