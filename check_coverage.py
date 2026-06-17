#!/usr/bin/env python
"""
Coverage checker for rpyforth.

Analyzes the implementation against:
  - Forth 2012 Core wordset
  - Classic Doug Bagley / Win32 Shootout benchmark requirements

Prints full coverage reports to stdout and writes:
  - FORTH2012_COVERAGE.md
  - SHOOTOUT_COVERAGE.md

See docs/SHOOTOUT_FULL_SUPPORT_PLAN.md for the full support roadmap.
"""

from __future__ import print_function

import os
import re

# Forth 2012 Core wordset (133 words) - Official ANS Forth 2012 standard
# Source: https://forth-standard.org/standard/core
FORTH_2012_CORE_WORDS = {
    "DUP", "DROP", "SWAP", "OVER", "ROT", "?DUP", "PICK",
    "2DUP", "2DROP", "2SWAP", "2OVER",
    ">R", "R>", "R@", "2>R", "2R>", "2R@",
    "+", "-", "*", "/", "MOD", "/MOD", "*/", "*/MOD",
    "1+", "1-", "2*", "2/",
    "ABS", "NEGATE", "MAX", "MIN",
    "M*", "UM*", "FM/MOD", "SM/REM", "UM/MOD",
    "=", "<", ">", "U<", "0=", "0<", "0>",
    "AND", "OR", "XOR", "INVERT", "LSHIFT", "RSHIFT",
    "!", "@", "C!", "C@", "+!", "2!", "2@",
    "CELL+", "CELLS", "CHAR+", "CHARS", "ALIGNED", "ALIGN",
    "HERE", "ALLOT", ",", "C,",
    ":", ";", "IMMEDIATE", "RECURSE",
    "[", "]", "LITERAL", "[']", "[CHAR]", "POSTPONE", "DOES>",
    "IF", "ELSE", "THEN", "BEGIN", "WHILE", "REPEAT", "UNTIL", "AGAIN",
    "DO", "LOOP", "+LOOP", "UNLOOP", "LEAVE", "I", "J", "EXIT",
    "BL", "S>D",
    "VARIABLE", "CONSTANT", "CREATE",
    "FIND", "EXECUTE", ">BODY",
    "EMIT", "TYPE", "CR", "SPACE", "SPACES", ".", "U.",
    ".\"", "S\"",
    "KEY", "ACCEPT",
    ">NUMBER", "BASE",
    "WORD", "CHAR", "COUNT",
    "SOURCE", ">IN",
    "<#", "#", "#S", "#>", "HOLD", "SIGN",
    "STATE", "QUIT", "ABORT", "ABORT\"",
    "EVALUATE", "ENVIRONMENT?",
    "FILL", "MOVE", "DECIMAL", "HEX",
    "'", "(", "DEPTH",
}

SHOOTOUT_EXTENSION_WORDS = {
    "CLI & timing": {"ARG", "ARGC", "ARGV", "UTIME", "CPUTIME"},
    "Memory & allocation": {"ALLOCATE", "FREE", "RESIZE"},
    "File I/O": {
        "READ-FILE", "READ-LINE", "WRITE-FILE",
        "OPEN-FILE", "CLOSE-FILE", "R/O", "R/W",
        "STDIN", "STDOUT", "STDERR",
    },
    "String / block copy": {"CMOVE", "CMOVE>"},
    "Control flow extensions": {
        "?DO", "CASE", "OF", "ENDOF", "ENDCASE", "RECURSIVE",
    },
    "Exceptions": {"CATCH", "THROW"},
    "Dynamic definitions": {"VALUE", "TO"},
    "Search order / dictionary": {
        "WORDLIST", "SEARCH-WORDLIST", "NEXTNAME",
        "GET-CURRENT", "SET-CURRENT", ">ORDER", "PREVIOUS",
        "WORDLIST-ID", "NAME>INT", "NAME>STRING", "LASTXT",
    },
    "Structures": {"STRUCT", "END-STRUCT", "FIELD", "CELL%", "%ALLOT"},
    "Floating extensions": {
        "FVARIABLE", "FCONSTANT", "F,", ">FLOAT", "F$",
        "SET-PRECISION", "PRECISION", "V*",
    },
    "Gforth libraries (Tier C)": {"REQUIRE", "INCLUDE", "OBJECTS", "TASKER", "GRAY"},
}

SHOOTOUT_BENCHMARKS = [
    {"id": "ack", "title": "Ackermann's function", "original": "ackermann.gforth",
     "path": "shootout/ack.fs", "phase": 0, "status": "supported",
     "required": {"RECURSIVE", "UTIME", "ARG"}},
    {"id": "ary", "title": "Array access", "original": "ary3.gforth",
     "path": "shootout/ary.fs", "phase": 0, "status": "supported",
     "required": {"ALLOCATE", "UTIME", "ARG"}},
    {"id": "fibo", "title": "Fibonacci", "original": "fibo.gforth",
     "path": "shootout/fibo.fs", "phase": 0, "status": "supported",
     "required": {"RECURSIVE", "UTIME", "ARG"}},
    {"id": "heap", "title": "Heapsort", "original": "heapsort.gforth",
     "path": "shootout/heap.fs", "phase": 0, "status": "supported",
     "required": {"ALLOCATE", "FVARIABLE", "SET-PRECISION", "UTIME", "ARG"}},
    {"id": "nestedloop", "title": "Nested loops", "original": "nestedloop.gforth",
     "path": "shootout/nestedloop.fs", "phase": 0, "status": "supported",
     "required": {"UTIME", "ARG"}},
    {"id": "sieve", "title": "Sieve of Eratosthenes", "original": "sieve.gforth",
     "path": "shootout/sieve.fs", "phase": 0, "status": "supported",
     "required": {"UTIME", "ARG"}},
    {"id": "hello", "title": "Hello world", "original": "hello.gforth",
     "path": "shootout/hello.fs", "phase": 1, "status": "missing", "required": set()},
    {"id": "sumcol", "title": "Sum a column of integers", "original": "sumcol.gforth",
     "path": "shootout/sumcol.fs", "phase": 1, "status": "missing",
     "required": {"READ-LINE", "STDIN", "ARG"}},
    {"id": "strcat", "title": "String concatenation", "original": "strcat.gforth",
     "path": "shootout/strcat.fs", "phase": 1, "status": "missing",
     "required": {"ALLOCATE", "RESIZE", "CMOVE>", "ARG"}},
    {"id": "reversefile", "title": "Reverse lines from stdin", "original": "reversefile.gforth",
     "path": "shootout/reversefile.fs", "phase": 1, "status": "missing",
     "required": {"READ-FILE", "WRITE-FILE", "ALLOCATE", "STDIN", "STDOUT", "ARG"}},
    {"id": "wc", "title": "Word / line / char count", "original": "wc.gforth",
     "path": "shootout/wc.fs", "phase": 1, "status": "missing",
     "required": {"READ-FILE", "STDIN", "CASE", "OF", "ENDOF", "ENDCASE", "ARG"}},
    {"id": "except", "title": "Exception handling", "original": "except.gforth",
     "path": "shootout/except.fs", "phase": 2, "status": "missing",
     "required": {"CATCH", "THROW", "ARG"}},
    {"id": "random", "title": "Random numbers", "original": "random.gforth",
     "path": "shootout/random.fs", "phase": 2, "status": "missing",
     "required": {"VALUE", "TO", "SET-PRECISION", "F$", "ARG"}},
    {"id": "hash", "title": "Hash table (int keys)", "original": "hash.gforth",
     "path": "shootout/hash.fs", "phase": 3, "status": "missing",
     "required": {"WORDLIST", "SEARCH-WORDLIST", "NEXTNAME",
                  "GET-CURRENT", "SET-CURRENT", "?DO", "ARG"}},
    {"id": "hash2", "title": "Hash table (two-level)", "original": "hash2.gforth",
     "path": "shootout/hash2.fs", "phase": 3, "status": "missing",
     "required": {"WORDLIST", "SEARCH-WORDLIST", "NEXTNAME",
                  "GET-CURRENT", "SET-CURRENT", ">ORDER", "PREVIOUS",
                  "NAME>INT", "NAME>STRING", "LASTXT", "?DO", "ARG"}},
    {"id": "spellcheck", "title": "Spell checker", "original": "spellcheck.gforth",
     "path": "shootout/spellcheck.fs", "phase": 3, "status": "missing",
     "required": {"OPEN-FILE", "READ-LINE", "WORDLIST", "SEARCH-WORDLIST",
                  "NEXTNAME", "GET-CURRENT", "SET-CURRENT", "STDIN", "ARG"}},
    {"id": "wordfreq", "title": "Word frequency count", "original": "wordfreq.gforth",
     "path": "shootout/wordfreq.fs", "phase": 3, "status": "missing",
     "required": {"WORDLIST", "SEARCH-WORDLIST", "NEXTNAME",
                  "GET-CURRENT", "SET-CURRENT", "READ-LINE", "STDIN",
                  "STRUCT", "END-STRUCT", "FIELD", "CELL%", "ARG"}},
    {"id": "lists", "title": "Linked list operations", "original": "lists.gforth",
     "path": "shootout/lists.fs", "phase": 4, "status": "missing",
     "required": {"STRUCT", "END-STRUCT", "FIELD", "CELL%", "%ALLOT", "?DO", "THROW", "ARG"}},
    {"id": "moments", "title": "Statistical moments", "original": "moments.gforth",
     "path": "shootout/moments.fs", "phase": 4, "status": "missing",
     "required": {"READ-LINE", "STDIN", ">FLOAT", "F,", "FLOATS", "SET-PRECISION", "F$", "ARG"}},
    {"id": "matrix", "title": "Matrix multiplication", "original": "matrix.gforth",
     "path": "shootout/matrix.fs", "phase": 5, "status": "missing",
     "required": {"V*", "POSTPONE", "ARG"}},
    {"id": "methcall", "title": "Method calls (OO)", "original": "methcall.gforth",
     "path": "shootout/methcall.fs", "phase": 6, "status": "out_of_scope",
     "required": {"REQUIRE", "OBJECTS", "ARG"}},
    {"id": "objinst", "title": "Object instantiation (OO)", "original": "objinst.gforth",
     "path": "shootout/objinst.fs", "phase": 6, "status": "out_of_scope",
     "required": {"REQUIRE", "OBJECTS", "ARG"}},
    {"id": "prodcons", "title": "Producer / consumer threads", "original": "prodcons.gforth",
     "path": "shootout/prodcons.fs", "phase": 6, "status": "out_of_scope",
     "required": {"REQUIRE", "TASKER", "ARG"}},
    {"id": "regexmatch", "title": "Regex / phone number scan", "original": "regexmatch.gforth",
     "path": "shootout/regexmatch.fs", "phase": 6, "status": "out_of_scope",
     "required": {"REQUIRE", "GRAY", "READ-FILE", "STDIN", "ARG"}},
]


BAR_WIDTH = 20
CORE_LABEL_WIDTH = 24
SHOOTOUT_LABEL_WIDTH = 22
SHOOTOUT_EXT_LABEL_WIDTH = 26


def coverage_bar(pct, width=BAR_WIDTH):
    if pct >= 100.0:
        filled = width
    elif pct <= 0.0:
        filled = 0
    else:
        filled = int(width * pct / 100.0)
        if filled == 0 and pct > 0.0:
            filled = 1
    return "#" * filled + "." * (width - filled)


def bar_line(label, impl, total, label_width=CORE_LABEL_WIDTH, suffix=""):
    pct = 100.0 * impl / total if total else 0.0
    line = "  %-{}s [%s] %5.1f%% (%d/%d)".format(label_width) % (
        label, coverage_bar(pct), pct, impl, total,
    )
    if suffix:
        line += "  %s" % suffix
    return line


def bar_line_pct(label, pct, label_width=CORE_LABEL_WIDTH, suffix=""):
    line = "  %-{}s [%s] %5.1f%%".format(label_width) % (
        label, coverage_bar(pct), pct,
    )
    if suffix:
        line += "  %s" % suffix
    return line


def missing_suffix(words, prefix="-"):
    if not words:
        return ""
    return "%s %s" % (prefix, ", ".join(words))


def extract_primitives_from_file(filepath):
    words = set()
    with open(filepath, "r") as f:
        content = f.read()
    words.update(re.findall(r'outer\.define_prim\("([^"]+)"', content))
    return words


def extract_special_words_from_outer(filepath):
    words = set()
    with open(filepath, "r") as f:
        content = f.read()
    for pattern in (
        r'if\s+tkey\s*==\s*"([^"]+)"',
        r'tkey\s*==\s*"([^"]+)"',
        r"if\s+t\s*==\s*'([^']+)'",
        r'if\s+t\s*==\s*"([^"]+)"',
    ):
        words.update(re.findall(pattern, content))
    return words


def collect_implemented_words(script_dir):
    implemented = set()
    for word in (
        extract_primitives_from_file(os.path.join(script_dir, "rpyforth", "primitives.py"))
        | extract_special_words_from_outer(os.path.join(script_dir, "rpyforth", "outer_interp.py"))
        | {":", ";"}
    ):
        implemented.add(word.upper())
    return implemented


def categorize_core_words():
    return {
        "Stack Manipulation": {
            "DUP", "DROP", "SWAP", "OVER", "ROT", "?DUP", "PICK",
            "2DUP", "2DROP", "2SWAP", "2OVER",
        },
        "Return Stack": {">R", "R>", "R@", "2>R", "2R>", "2R@"},
        "Arithmetic": {
            "+", "-", "*", "/", "MOD", "/MOD", "*/", "*/MOD",
            "1+", "1-", "2*", "2/",
            "ABS", "NEGATE", "MAX", "MIN",
            "M*", "UM*", "FM/MOD", "SM/REM", "UM/MOD",
        },
        "Comparison": {"=", "<", ">", "U<", "0=", "0<", "0>", "0<>"},
        "Logical & Bitwise": {"AND", "OR", "XOR", "INVERT", "LSHIFT", "RSHIFT"},
        "Memory Access": {
            "!", "@", "C!", "C@", "+!", "2!", "2@",
            "CELL+", "CELLS", "CHAR+", "CHARS", "ALIGNED", "ALIGN",
        },
        "Data Space": {"HERE", "ALLOT", ",", "C,"},
        "Compilation": {
            ":", ";", "IMMEDIATE", "RECURSE", "[", "]", "LITERAL",
            "[']", "[CHAR]", "POSTPONE", "DOES>",
        },
        "Control Flow": {
            "IF", "ELSE", "THEN", "BEGIN", "WHILE", "REPEAT", "UNTIL", "AGAIN",
            "DO", "LOOP", "+LOOP", "UNLOOP", "LEAVE", "I", "J", "EXIT",
        },
        "Variables & Constants": {"VARIABLE", "CONSTANT", "CREATE"},
        "Dictionary": {"FIND", "EXECUTE", ">BODY"},
        "I/O": {
            "EMIT", "TYPE", "CR", "SPACE", "SPACES", ".", "U.",
            '.\"', "S\"",
            "KEY", "ACCEPT",
        },
        "Number Conversion": {">NUMBER", "BASE", "DECIMAL", "HEX"},
        "Parsing": {"WORD", "CHAR", "COUNT"},
        "Source Input": {"SOURCE", ">IN"},
        "Pictured Numeric Output": {"<#", "#", "#S", "#>", "HOLD", "SIGN"},
        "System": {
            "STATE", "QUIT", "ABORT", "ABORT\"",
            "EVALUATE", "ENVIRONMENT?", "FILL", "MOVE", "DEPTH", "BL", "S>D",
        },
        "Special": {"'", "("},
    }


def generate_core_report(implemented_core, missing, categories):
    total = len(FORTH_2012_CORE_WORDS)
    impl_count = len(implemented_core)
    coverage_pct = 100.0 * impl_count / total if total else 0.0
    lines = [
        "Forth 2012 Core Coverage",
        "%d/%d (%.1f%%)%s" % (
            impl_count, total, coverage_pct,
            missing_suffix(sorted(missing), "  missing:"),
        ),
        bar_line("Overall", impl_count, total),
    ]
    for category, words in sorted(categories.items()):
        cat_total = len(words)
        cat_impl = len(words & implemented_core)
        miss_words = sorted(words & missing)
        lines.append(bar_line(
            category, cat_impl, cat_total,
            suffix=missing_suffix(miss_words),
        ))
    return "\n".join(lines)


def all_shootout_extension_words():
    words = set()
    for group in SHOOTOUT_EXTENSION_WORDS.values():
        words |= group
    return words


def benchmark_readiness(benchmark, implemented):
    required = set(w.upper() for w in benchmark["required"])
    if not required:
        return 100.0, []
    missing = sorted(required - implemented)
    pct = 100.0 * (len(required) - len(missing)) / len(required)
    return pct, missing


def sync_benchmark_file_status(repo_root):
    synced = []
    for bench in SHOOTOUT_BENCHMARKS:
        entry = dict(bench)
        path = os.path.join(repo_root, entry["path"])
        if entry["status"] == "supported" and not os.path.isfile(path):
            entry["status"] = "missing_source"
        synced.append(entry)
    return synced


def generate_shootout_report(implemented, repo_root):
    benchmarks = sync_benchmark_file_status(repo_root)
    extension_words = all_shootout_extension_words()
    ext_impl = extension_words & implemented
    supported = [b for b in benchmarks if b["status"] == "supported"]
    missing_bench = [b for b in benchmarks if b["status"] == "missing"]
    tier_c = [b for b in benchmarks if b["status"] == "out_of_scope"]

    lines = [
        "Classic Shootout Coverage  plan: docs/SHOOTOUT_FULL_SUPPORT_PLAN.md",
        "%d/23 supported  %d missing  %d tier-C  ext %d/%d"
        % (len(supported), len(missing_bench), len(tier_c), len(ext_impl), len(extension_words)),
        bar_line("Programs", len(supported), 23),
        bar_line("Extension words", len(ext_impl), len(extension_words)),
        "Benchmarks",
    ]
    status_abbr = {
        "supported": "ok",
        "missing": "todo",
        "out_of_scope": "tier",
        "missing_source": "nosrc",
    }
    for bench in sorted(benchmarks, key=lambda b: (b["phase"], b["id"])):
        pct, miss = benchmark_readiness(bench, implemented)
        label = "%s P%d %s" % (
            status_abbr.get(bench["status"], bench["status"][:4]),
            bench["phase"],
            bench["id"],
        )
        lines.append(bar_line_pct(
            label, pct,
            label_width=SHOOTOUT_LABEL_WIDTH,
            suffix=missing_suffix(miss),
        ))

    lines.append("Extension words")
    for category, words in sorted(SHOOTOUT_EXTENSION_WORDS.items()):
        impl = sorted(words & implemented)
        miss = sorted(words - implemented)
        lines.append(bar_line(
            category, len(impl), len(words),
            label_width=SHOOTOUT_EXT_LABEL_WIDTH,
            suffix=missing_suffix(miss),
        ))

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    implemented = collect_implemented_words(script_dir)

    print("Analyzing rpyforth (%d words)..." % len(implemented))

    categories = categorize_core_words()
    core_impl = implemented & FORTH_2012_CORE_WORDS
    core_missing = FORTH_2012_CORE_WORDS - implemented

    core_report = generate_core_report(core_impl, core_missing, categories)
    shootout_report = generate_shootout_report(implemented, script_dir)

    core_report_path = os.path.join(script_dir, "FORTH2012_COVERAGE.md")
    shootout_report_path = os.path.join(script_dir, "SHOOTOUT_COVERAGE.md")
    with open(core_report_path, "w") as f:
        f.write(core_report)
    with open(shootout_report_path, "w") as f:
        f.write(shootout_report)
    print("Wrote %s and %s" % (core_report_path, shootout_report_path))
    print()
    print(core_report)
    print()
    print(shootout_report)


if __name__ == "__main__":
    main()
