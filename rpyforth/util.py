from rpython.rlib.jit import unroll_safe

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
        while i < n and depth > 0:
            if line[i] == '(':
                depth += 1
            elif line[i] == ')':
                depth -= 1
            i += 1
        if depth > 0:
            return result, depth

    while i < n:
        ch = line[i]

        # Handle backslash comment - rest of line is ignored
        if ch == '\\':
            # Check if it's actually a backslash comment (needs space before or at start)
            if i == 0 or line[i-1] in ' \t\n\r\v\f':
                break  # Skip rest of line

        # Handle parenthetical comment. '(' opens a comment only when it stands
        # as its own word: whitespace (or start) before it AND whitespace (or
        # end of line) after it. This leaves paren-named words like (checkTime)
        # or (mv) -- where '(' is glued to the name -- intact.
        if ch == '(':
            before_ok = i == 0 or line[i-1] in ' \t\n\r\v\f'
            after_ok = i + 1 >= n or line[i+1] in ' \t\n\r\v\f'
            if before_ok and after_ok:
                # Find matching ) with nesting support
                i += 1
                depth = 1
                while i < n and depth > 0:
                    if line[i] == '(':
                        depth += 1
                    elif line[i] == ')':
                        depth -= 1
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
    # Remove comments before tokenization
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
