"""Lexer and parser for the Factor-compatible rpyfactor subset (Phase A).

Supported:
  : name ( in -- out ) body ;
  [ ... ] quotations
  { ... } array literals (ints / nested arrays / quotations)
  ! line comments
  Stack-effect forms ( ... -- ... ) are accepted and ignored
"""

from rpyfactor.program import (
    LitInt, LitBool, LitString, LitSymbol, LitQuot, CallWord,
)
from rpyfactor.values import W_List, W_Int, W_Quotation


class ParseError(Exception):
    pass


def _at(text, i, prefix):
    n = len(prefix)
    return i + n <= len(text) and text[i:i + n] == prefix


def _is_punct(tok, val):
    return tok[0] == "punct" and tok[1] == val


def tokenize(text):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "!":
            # Factor: "!" starts a line comment, but "!=" is a word.
            if _at(text, i, "!="):
                tokens.append(("word", "!="))
                i += 2
                continue
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "(":
            # Nestable comment / stack-effect form: skip until matching ')'.
            depth = 1
            i += 1
            while i < n and depth > 0:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            continue
        if _at(text, i, "--"):
            # Bare -- outside ( ) should not appear; treat as word-ish skip.
            tokens.append(("word", "--"))
            i += 2
            continue
        if c in "[]{};:":
            tokens.append(("punct", c))
            i += 1
            continue
        if c == '"':
            i += 1
            start = i
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i >= n:
                raise ParseError("unterminated string")
            tokens.append(("string", text[start:i]))
            i += 1
            continue
        if c == "'":
            # Factor word quoting: ' word  -> push the word as a symbol
            i += 1
            while i < n and text[i].isspace():
                i += 1
            start = i
            while i < n and not text[i].isspace() and text[i] not in "[]{};:!()\"":
                i += 1
            if start == i:
                raise ParseError("unterminated word quote")
            tokens.append(("char", text[start:i]))
            continue
        start = i
        while i < n:
            if text[i].isspace():
                break
            if text[i] in "[]{};:!()\"":
                break
            if text[i] == "!":
                break
            i += 1
        word = text[start:i]
        if not word:
            raise ParseError("empty token at %d" % start)
        tokens.append(("word", word))
    return tokens


def _parse_items(tokens, pos, terminators):
    """Parse a sequence of program items until a terminator punct is seen."""
    items = []
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if tok[0] == "punct" and tok[1] in terminators:
            return items
        if _is_punct(tok, "["):
            pos[0] += 1
            body = _parse_items(tokens, pos, "]")
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "]"):
                raise ParseError("expected ]")
            pos[0] += 1
            items.append(LitQuot(body))
            continue
        if _is_punct(tok, "{"):
            pos[0] += 1
            elems = _parse_array_elems(tokens, pos)
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "}"):
                raise ParseError("expected }")
            pos[0] += 1
            # Array literal: push as a CallWord that materializes? Better as
            # a special literal. Encode as LitQuot of a synthetic push via
            # wrapping in a list-building form. Use LitQuot that is actually
            # a list value through item_to_value — add LitArray.
            from rpyfactor.program import LitArray
            items.append(LitArray(elems))
            continue
        if _is_punct(tok, ":"):
            raise ParseError("nested : definition not allowed here")
        kind = tok[0]
        val = tok[1]
        if kind == "string":
            items.append(LitString(val))
            pos[0] += 1
            continue
        if kind == "char":
            items.append(LitSymbol(val))
            pos[0] += 1
            continue
        if kind == "word":
            pos[0] += 1
            if val == "t" or val == "true":
                items.append(LitBool(True))
            elif val == "f" or val == "false":
                items.append(LitBool(False))
            elif _is_int_token(val):
                items.append(LitInt(int(val)))
            else:
                items.append(CallWord(val))
            continue
        raise ParseError("unexpected token: %s %s" % (tok[0], tok[1]))
    return items


def _parse_array_elems(tokens, pos):
    """Parse array elements into a list of W_Value-ready program items.

    Elements may be ints, bools, nested arrays, or quotations. They are
    stored as program items and converted when the LitArray is pushed.
    """
    elems = []
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if _is_punct(tok, "}"):
            return elems
        if _is_punct(tok, "{"):
            pos[0] += 1
            nested = _parse_array_elems(tokens, pos)
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "}"):
                raise ParseError("expected }")
            pos[0] += 1
            from rpyfactor.program import LitArray
            elems.append(LitArray(nested))
            continue
        if _is_punct(tok, "["):
            pos[0] += 1
            body = _parse_items(tokens, pos, "]")
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "]"):
                raise ParseError("expected ]")
            pos[0] += 1
            elems.append(LitQuot(body))
            continue
        kind = tok[0]
        val = tok[1]
        if kind == "string":
            elems.append(LitString(val))
            pos[0] += 1
            continue
        if kind == "word":
            pos[0] += 1
            if val == "t" or val == "true":
                elems.append(LitBool(True))
            elif val == "f" or val == "false":
                elems.append(LitBool(False))
            elif _is_int_token(val):
                elems.append(LitInt(int(val)))
            else:
                elems.append(CallWord(val))
            continue
        raise ParseError("bad array element: %s %s" % (tok[0], tok[1]))
    return elems


def _is_int_token(val):
    if not val:
        return False
    if val[0] == "-" and len(val) > 1:
        return val[1:].isdigit()
    return val.isdigit()


def parse_definitions(text):
    """Parse `: name ... ;` blocks; return dict name -> program list."""
    tokens = tokenize(text)
    defs = {}
    pos = [0]
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if _is_punct(tok, ":"):
            pos[0] += 1
            if pos[0] >= len(tokens) or tokens[pos[0]][0] != "word":
                raise ParseError(": expects a name")
            name = tokens[pos[0]][1]
            pos[0] += 1
            body = _parse_items(tokens, pos, ";")
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], ";"):
                raise ParseError(": expects terminating ;")
            pos[0] += 1
            defs[name] = body
        else:
            pos[0] += 1
    return defs


def parse_source(text):
    """Return (definitions dict, main program items)."""
    defs = parse_definitions(text)
    tokens = tokenize(text)
    pos = [0]
    program = []
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if _is_punct(tok, ":"):
            pos[0] += 1
            if pos[0] < len(tokens) and tokens[pos[0]][0] == "word":
                pos[0] += 1
            _parse_items(tokens, pos, ";")
            if pos[0] < len(tokens) and _is_punct(tokens[pos[0]], ";"):
                pos[0] += 1
            continue
        if _is_punct(tok, ";"):
            pos[0] += 1
            continue
        if _is_punct(tok, "]") or _is_punct(tok, "}"):
            pos[0] += 1
            continue
        chunk = _parse_items(tokens, pos, ";:")
        program.extend(chunk)
        if pos[0] < len(tokens) and _is_punct(tokens[pos[0]], ";"):
            pos[0] += 1
    return defs, program
