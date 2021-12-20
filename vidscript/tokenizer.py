from typing import List, Tuple

# This parser treats commas as whitespace, so this is a special helper to manage that
def is_whitespace(s:str) -> bool:
    for c in s:
        if c <= ' ' or c == ',':
            return True
    return False

# Finds the next whitespace character in a string (starting with beg)
def find_whitespace(s:str, beg:int=0) -> int:
    for i in range(beg, len(s)):
        if is_whitespace(s[i]):
            return i
    return -1

# Removes whitespace from the start of a string
def ltrim(s:str) -> str:
    for i in range(len(s)):
        if not is_whitespace(s[i]):
            return s[i:]
    return ''

# Removes whitespace from the end of a string
def rtrim(s:str) -> str:
    for i in reversed(range(len(s))):
        if not is_whitespace(s[i]):
            return s[:i+1]
    return ''


class Tokenizer():
    # s is the string to tokenize
    # line_nums indicates where line-numbers begin. (First = string position, Second = line number)
    def __init__(self, s:str, line_nums:List[Tuple[int,int]]) -> None:
        self.s = s
        self.pos = 0
        self.line_nums = line_nums

    def advance(self, chars:int) -> str:
        tmp = self.s[self.pos:self.pos+chars]
        self.pos += chars
        return tmp

    def shorten(self, chars:int) -> None:
        self.s = self.s[:-chars]

    def remaining(self) -> int:
        return len(self.s) - self.pos

    def peek(self) -> str:
        return self.s[self.pos:]

    # Divides everything before index and everything including index and after into two Tokenizers
    def split(self, index:int) -> Tuple['Tokenizer', 'Tokenizer']:
        s1 = self.s[self.pos:self.pos+index]
        s2 = self.s[self.pos+index:]
        line1, _ = self.line_col()
        line2, _ = self.line_col(index)
        line_nums1 = [(0, line1)]
        line_nums2 = [(0, line2)]
        for pos, line in self.line_nums:
            if pos < self.pos:
                pass
            elif pos < self.pos + index:
                if line != line_nums1[-1][1]:
                    line_nums1.append((pos - self.pos, line))
            else:
                if line != line_nums2[-1][1]:
                    line_nums2.append((pos - self.pos - index, line))
        left = Tokenizer(s1, line_nums1)
        right = Tokenizer(s2, line_nums2)
        return left, right

    def find_ws(self) -> int:
        i = 0
        while True:
            if self.pos + i >= len(self.s):
                return -1
            if is_whitespace(self.s[self.pos + i]):
                return i
            i += 1

    def find_ws_or_eq(self) -> int:
        i = 0
        while True:
            if self.pos + i >= len(self.s):
                return -1
            if is_whitespace(self.s[self.pos + i]) or self.s[self.pos + i] == '=':
                return i
            i += 1

    def find_non_ws(self) -> int:
        i = 0
        while True:
            if self.pos + i >= len(self.s):
                return -1
            if not is_whitespace(self.s[self.pos + i]):
                return i
            i += 1

    def find_any_of(self, chars:str) -> int:
        i = 0
        while True:
            if self.pos + i >= len(self.s):
                return -1
            if chars.find(self.s[self.pos + i]) >= 0:
                return i
            i += 1

    def skip_ws(self) -> None:
        n = self.find_non_ws()
        if n >= 0:
            self.advance(n)
        else:
            pos = len(self.s)

    def rtrim(self) -> None:
        i = len(self.s)
        while i > 0 and is_whitespace(self.s[i - 1]):
            i -= 1
        self.s = self.s[:i]

    def line_col(self, offset:int=0) -> Tuple[int, int]:
        start = 0
        line = 0
        for i in range(len(self.line_nums)):
            if self.pos + offset >= self.line_nums[i][0]:
                start = self.line_nums[i][0]
                line = self.line_nums[i][1]
            else:
                break
        return line, self.pos - start

    def err(self, msg:str) -> None:
        line, _ = self.line_col()
        raise ValueError(f'Parse error at line {line + 1}: {msg}')


# Merges the fragments into a single string and a list of tuples to indicate where lines begin
def from_fragments(fragments:List[Tuple[str,int]]) -> Tokenizer:
    s = ''.join([ frag[0] for frag in fragments ])
    line_nums:List[Tuple[int,int]] = [] # First = string position, Second = line number
    sum_len = 0
    for frag in fragments:
        line_nums.append((sum_len, frag[1]))
        sum_len += len(frag[0])
    return Tokenizer(s, line_nums)

# Merges the fragments into a single string and a list of tuples to indicate where lines begin
def from_lines(lines:List[str], line_nums:List[int]) -> Tokenizer:
    s = ''.join(lines)
    l_nums:List[Tuple[int,int]] = [] # First = string position, Second = line number
    sum_len = 0
    for i in range(min(len(lines), len(line_nums))):
        l_nums.append((sum_len, line_nums[i]))
        sum_len += len(lines[i])
    return Tokenizer(s, l_nums)
