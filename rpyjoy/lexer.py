"""Lexer and parser for the Joy-compatible rpyjoy subset."""

from rpyjoy.program import (
    LitInt, LitBool, LitString, LitSymbol, LitQuot, CallWord,
)


class ParseError(Exception):
    pass


def _at(text, i, prefix):
    n = len(prefix)
    return i + n <= len(text) and text[i:i + n] == prefix


def _is_punct(tok, val):
    return tok[0] == "punct" and tok[1] == val


def _skip_comment(text, i):
    if i + 1 < len(text) and text[i:i + 2] == "(*":
        depth = 1
        i += 2
        while i < len(text) and depth > 0:
            if i + 1 < len(text) and text[i:i + 2] == "(*":
                depth += 1
                i += 2
            elif i + 1 < len(text) and text[i:i + 2] == "*)":
                depth -= 1
                i += 2
            else:
                i += 1
        return i
    return -1


def tokenize(text):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        skipped = _skip_comment(text, i)
        if skipped >= 0:
            i = skipped
            continue
        if _at(text, i, "=="):
            tokens.append(("punct", "=="))
            i += 2
            continue
        if c in "[]().":
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
            i += 1
            if i >= n:
                raise ParseError("unterminated char literal")
            tokens.append(("char", text[i]))
            i += 1
            continue
        start = i
        while i < n:
            if text[i].isspace():
                break
            if text[i] in "[]().":
                break
            if _at(text, i, "(*"):
                break
            if _at(text, i, "=="):
                break
            if text[i] == '"':
                break
            if text[i] == "'":
                break
            i += 1
        word = text[start:i]
        if not word:
            raise ParseError("empty token at %d" % start)
        tokens.append(("word", word))
    return tokens


def _parse_items(tokens, pos):
    items = []
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if _is_punct(tok, "]"):
            return items
        if _is_punct(tok, "."):
            return items
        if _is_punct(tok, "["):
            pos[0] += 1
            body = _parse_items(tokens, pos)
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "]"):
                raise ParseError("expected ]")
            pos[0] += 1
            items.append(LitQuot(body))
            continue
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
            if val == "true":
                items.append(LitBool(True))
            elif val == "false":
                items.append(LitBool(False))
            elif val.lstrip("-").isdigit():
                items.append(LitInt(int(val)))
            else:
                items.append(CallWord(val))
            continue
        if _is_punct(tok, "=="):
            raise ParseError("unexpected ==")
        raise ParseError("unexpected token: %s %s" % (tok[0], tok[1]))
    return items


def parse_program(text):
    tokens = tokenize(text)
    pos = [0]
    return _parse_items(tokens, pos)


def parse_definitions(text):
    """Parse DEFINE blocks; return dict name -> program tuple."""
    tokens = tokenize(text)
    defs = {}
    pos = [0]
    while pos[0] < len(tokens):
        tok = tokens[pos[0]]
        if tok[0] == "word" and tok[1] == "DEFINE":
            pos[0] += 1
            if pos[0] >= len(tokens) or tokens[pos[0]][0] != "word":
                raise ParseError("DEFINE expects a name")
            name = tokens[pos[0]][1]
            pos[0] += 1
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "=="):
                raise ParseError("DEFINE expects ==")
            pos[0] += 1
            body = _parse_items(tokens, pos)
            if pos[0] >= len(tokens) or not _is_punct(tokens[pos[0]], "."):
                raise ParseError("DEFINE expects terminating .")
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
        if tok[0] == "word" and tok[1] == "DEFINE":
            pos[0] += 1
            if pos[0] < len(tokens):
                pos[0] += 1
            if pos[0] < len(tokens) and _is_punct(tokens[pos[0]], "=="):
                pos[0] += 1
            _parse_items(tokens, pos)
            if pos[0] < len(tokens) and _is_punct(tokens[pos[0]], "."):
                pos[0] += 1
            continue
        if _is_punct(tok, "."):
            pos[0] += 1
            continue
        if _is_punct(tok, "]"):
            pos[0] += 1
            continue
        chunk = _parse_items(tokens, pos)
        program.extend(chunk)
        if pos[0] < len(tokens) and _is_punct(tokens[pos[0]], "."):
            pos[0] += 1
    return defs, program
