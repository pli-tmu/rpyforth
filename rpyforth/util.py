from rpython.rlib.jit import unroll_safe


def _is_space(ch):
    return (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or
            ch == '\v' or ch == '\f')


# Words that open a string literal terminated by '"'. Inside such a literal a
# backslash or paren is ordinary string content, not a comment, so comment
# removal must copy the literal verbatim up to the closing quote.
_QUOTE_STRING_WORDS = {
    'S"': True, 'C"': True, '."': True, 'ABORT"': True, 'S\\"': True,
}


def _delim_ahead(line, we, delim):
    """True if the closing delimiter appears at/after position we. we is the end
    of the opening word (the following char is the single delimiter space)."""
    j = we
    n = len(line)
    while j < n:
        if line[j] == delim:
            return True
        j += 1
    return False


def _quote_word_upper(word):
    out = ''
    for i in range(len(word)):
        o = ord(word[i])
        if o >= 97 and o <= 122:
            out += chr(o - 32)
        else:
            out += word[i]
    return out


def _prev_word_is_postpone(result):
    """True if the last whitespace-delimited word emitted so far is POSTPONE or
    [COMPILE] (case-insensitive). Used so `POSTPONE (` names the comment word
    rather than opening a comment. Builds the last word char by char to stay
    clear of RPython's non-negative-slice-bound requirement."""
    n = len(result)
    j = n - 1
    while j >= 0 and _is_space(result[j]):
        j -= 1
    # result[k] for k in (j .. first space below j] is the last word, reversed.
    word = ''
    while j >= 0 and not _is_space(result[j]):
        ch = result[j]
        o = ord(ch)
        if o >= 97 and o <= 122:
            ch = chr(o - 32)
        word = ch + word
        j -= 1
    return word == 'POSTPONE' or word == '[COMPILE]'


@unroll_safe
def to_upper(s):
    out = ''
    for i in range(len(s)):
        o = ord(s[i])
        if o >= 97 and o <= 122: # a-z
            out += chr(o - 32)
        else:
            out += s[i]
    return out

@unroll_safe
def remove_comments_stateful(line, depth):
    """Remove Forth comments from a line, carrying paren-comment nesting depth
    across lines.

    Handles two types:
    - \ comment: backslash to end of line
    - ( comment ): parenthetical comment, which may span multiple lines when a
      file is INCLUDEd (gforth behaviour). ``depth`` is the number of open '('
      comments inherited from previous lines.

    Returns (result, out_depth): the line with comments removed and the paren
    depth still open at end of line.
    """
    result = ''
    i = 0
    n = len(line)
    # If we start already inside a multi-line paren comment, consume characters
    # (tracking nesting) until the matching ')' closes it, then fall through to
    # normal processing of the remainder.
    if depth > 0:
        # ANS ( does not nest: the first ')' closes the comment.
        while i < n and depth > 0:
            if line[i] == ')':
                depth = 0
            i += 1
        if depth > 0:
            return result, depth

    while i < n:
        ch = line[i]
        at_word_start = i == 0 or _is_space(line[i-1])

        # At a word boundary, check whether this word opens a string literal
        # ('"'-terminated, e.g. S" ." ABORT") or a .( print. Inside such a literal
        # a '\' or '(' is ordinary content, so copy it verbatim to the closing
        # delimiter rather than treating it as a comment.
        if at_word_start and not _is_space(ch):
            we = i
            while we < n and not _is_space(line[we]):
                we += 1
            word = line[i:we]
            wu = _quote_word_upper(word)
            if wu == '.(' and _delim_ahead(line, we, ')'):
                # Copy the word, then everything up to and including the ')'.
                while i < n and line[i] != ')':
                    result += line[i]
                    i += 1
                if i < n:  # include the ')'
                    result += line[i]
                    i += 1
                continue
            # Only treat S" ." ABORT" etc as a string opener when a closing quote
            # actually follows on this line. Without one the token is plain data
            # (lexex lexinput.fth: `symbol ."` names the Forth word ." as a lexer
            # keyword), so let normal '\' comment stripping apply.
            if wu in _QUOTE_STRING_WORDS and _delim_ahead(line, we, '"'):
                # Copy the word plus one delimiter space, then the string body up
                # to and including the closing '"'.
                j = we
                result += word
                if j < n:  # the single delimiter space after the word
                    result += line[j]
                    j += 1
                while j < n and line[j] != '"':
                    result += line[j]
                    j += 1
                if j < n:  # include the closing quote
                    result += line[j]
                    j += 1
                i = j
                continue

        if ch == '\\':
            # A backslash comment needs a space before it or line start.
            if at_word_start:
                break

        # Handle parenthetical comment. '(' opens a comment only when it stands
        # as its own word: whitespace (or start) before it AND whitespace (or
        # end of line) after it. This leaves paren-named words like (checkTime)
        # or (mv) -- where '(' is glued to the name -- intact.
        if ch == '(':
            before_ok = at_word_start
            after_ok = i + 1 >= n or line[i+1] in ' \t\n\r\v\f'
            # POSTPONE ( and [COMPILE] ( name the comment word '(' rather than
            # opening a comment (gforth compat/assert.fs: `POSTPONE (`). Preserve
            # the '(' token in that case so the compiler can look it up.
            if before_ok and after_ok and _prev_word_is_postpone(result):
                result += ch
                i += 1
                continue
            if before_ok and after_ok:
                # ANS ( does not nest: scan to the first ')'.
                i += 1
                depth = 1
                while i < n and depth > 0:
                    if line[i] == ')':
                        depth = 0
                    i += 1
                if depth > 0:
                    # Unterminated on this line: the comment continues onto the
                    # next line (only meaningful inside an INCLUDEd file).
                    return result, depth
                continue

        result += ch
        i += 1

    return result, 0

@unroll_safe
def remove_comments(line):
    """Remove Forth comments from a single line (no cross-line paren state)."""
    result, _ = remove_comments_stateful(line, 0)
    return result

@unroll_safe
def split_whitespace_stateful(line, depth):
    """Split line into tokens, carrying paren-comment depth across lines.

    Returns (tokens, out_depth)."""
    line, depth = remove_comments_stateful(line, depth)
    res = _tokenize(line)
    return res, depth

@unroll_safe
def split_whitespace(line):
    """Split line into tokens, removing Forth comments first."""
    line = remove_comments(line)
    return _tokenize(line)

@unroll_safe
def _tokenize(line):

    res = []
    cur = ''
    for i in range(len(line)):
        ch = line[i]
        if ch == ' ' or ch == '\n' or ch == '\t' or ch == '\r' or \
           ch == '\v' or ch == '\f':
            if cur != '':
                res.append(cur)
                cur = ''
            continue
        cur += ch
    if cur != '':
        res.append(cur)
    return res


def digit_to_char(d):
    if d < 10:
        return chr(ord('0') + d)
    else:
        return chr(ord('A') + (d - 10))
