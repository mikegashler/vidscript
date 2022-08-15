import abc
from typing import Dict, Any, Optional, Tuple, List, cast, Union
import math
from . import tokenizer

ParamType = Union[float,str,Dict[str,Any]]

# Returns an integer representing the class of a character
def char_class_table_maker(n:int) -> int:
    if n < 65:
        if n < 33:
            return 0 # whitespace
        elif n < 48:
            if 40 <= n <= 41: # '(', ')'
                return 2 # parens
            elif n == 44: # ','
                return 0 # whitespace
            else:
                return 1 # symbol
        else:
            if n < 58: # digits
                return 3 # alpha-num
            else:
                return 1 # symbol
    else:
        if n < 97:
            if n < 91:
                return 3 # alpha-num
            elif n == 95: # '_'
                return 3 # alpha-num
            else:
                return 1 # symbol
        elif n < 123:
            return 3 # alpha-num
        elif n < 128:
            return 1 # symbol
        else:
            return 3 # alpha-num

char_class_table = { n:char_class_table_maker(n) for n in range(128) }

# Returns the char_class of any character (0=whitespace, 1=symbol, 2=parens, 3=alpha-numeric)
def char_class(c:str) -> int:
    n = ord(c)
    return char_class_table[n] if n < 128 else 3

# Operators
class Expr(abc.ABC):
    def __init__(self, params:List['Expr']) -> None:
        self.p = params

    @abc.abstractmethod
    def eval(self, locals:Dict[str,float]) -> float:
        raise ValueError('Abstract method was called!')

    def check_variables(self, line:int, params:Dict[str,ParamType], locals:Dict[str,'Expr']) -> None:
        for p in self.p:
            p.check_variables(line, params, locals)

    def __str__(self) -> str:
        s = str(type(self))
        dot = s.find('.')
        last_apost = s.rfind('\'')
        return f'{s[dot+1:last_apost]}({str(self.p)})'

class ExprConstant(Expr):
    def __init__(self, val:float) -> None:
        super().__init__([])
        self.val = val

    def eval(self, locals:Dict[str,float]) -> float:
        return self.val

    def __str__(self) -> str:
        return str(self.val)

class ExprVariable(Expr):
    def __init__(self, name:str) -> None:
        super().__init__([])
        self.name = name

    def eval(self, locals:Dict[str,float]) -> float:
        if not self.name in locals:
            raise ValueError(f'Error: Cannot resolve identifier {self.name}')
        return locals[self.name]

    def check_variables(self, line:int, params:Dict[str,ParamType], locals:Dict[str,Expr]) -> None:
        if self.name in params or self.name in locals:
            pass
        else:
            raise ValueError(f'Reference error in the expression that begins on line {line}: {self.name} not found')

    def __str__(self) -> str:
        return f'{self.name}'

class ExprStr(Expr):
    def __init__(self, name:str) -> None:
        super().__init__([])
        self.name = name

    def eval(self, locals:Dict[str,float]) -> float:
        return cast(float, self.name) # This is a deliberately bogus cast because ExprStr is a special case

    def __str__(self) -> str:
        return f'str({self.name})'

class ExprExponent(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return float(self.p[0].eval(locals) ** self.p[1].eval(locals))

    def __str__(self) -> str:
        return f'({self.p[0]} ^ {self.p[1]})'

class ExprMultiply(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return self.p[0].eval(locals) * self.p[1].eval(locals)

    def __str__(self) -> str:
        return f'({self.p[0]} * {self.p[1]})'

class ExprDivide(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        denom = self.p[1].eval(locals)
        return self.p[0].eval(locals) / denom if denom != 0 else 0.000000001

    def __str__(self) -> str:
        return f'({self.p[0]} / {self.p[1]})'

class ExprModulus(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return self.p[0].eval(locals) % self.p[1].eval(locals)

    def __str__(self) -> str:
        return f'({self.p[0]} % {self.p[1]})'

class ExprAdd(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return self.p[0].eval(locals) + self.p[1].eval(locals)

    def __str__(self) -> str:
        return f'({self.p[0]} + {self.p[1]})'

class ExprSubtract(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return self.p[0].eval(locals) - self.p[1].eval(locals)

    def __str__(self) -> str:
        return f'({self.p[0]} - {self.p[1]})'

class ExprAbsolute(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return abs(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'abs({self.p[0]})'

class ExprArcCosine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.acos(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'acos({self.p[0]})'

class ExprArcCosh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.acosh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'cosh({self.p[0]})'

class ExprArcSine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.asin(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'asin({self.p[0]})'

class ExprArcSinh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.asinh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'asinh({self.p[0]})'

class ExprArcTan(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.atan(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'atan({self.p[0]})'

class ExprArcTanh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.atanh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'atanh({self.p[0]})'

class ExprCeil(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return float(math.ceil(self.p[0].eval(locals)))

    def __str__(self) -> str:
        return f'ceil({self.p[0]})'

class ExprClip(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return max(min(self.p[0].eval(locals), self.p[2].eval(locals)), self.p[1].eval(locals))

    def __str__(self) -> str:
        return f'clip({self.p[0]}, {self.p[1]}, {self.p[2]})'

class ExprCosine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.cos(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'cos({self.p[0]})'

class ExprCosh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.cosh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'cosh({self.p[0]})'

class ExprErf(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.erf(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'erf({self.p[0]})'

class ExprFloor(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return float(math.floor(self.p[0].eval(locals)))

    def __str__(self) -> str:
        return f'floor({self.p[0]})'

class ExprGamma(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.gamma(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'gamma({self.p[0]})'

class ExprLogGamma(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.lgamma(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'lgamma({self.p[0]})'

class ExprLog(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.log(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'log({self.p[0]})'

class ExprMax(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return max([x.eval(locals) for x in self.p])

    def __str__(self) -> str:
        return f'max({", ".join([str(t) for t in self.p])})'

class ExprMin(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return min([x.eval(locals) for x in self.p])

    def __str__(self) -> str:
        return f'min({", ".join([str(t) for t in self.p])})'

class ExprSign(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.copysign(1., self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'sign({self.p[0]})'

class ExprSine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.sin(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'sin({self.p[0]})'

class ExprSinh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.sinh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'sinh({self.p[0]})'

class ExprSqrt(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.sqrt(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'sqrt({self.p[0]})'

class ExprTan(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.tan(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'tan({self.p[0]})'

class ExprTanh(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.tanh(self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'tanh({self.p[0]})'

class ExprTrans(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        a = self.p[0].eval(locals)
        b = max(1e-6, self.p[1].eval(locals))
        if a <= -0.5 * b:
            return self.p[2].eval(locals)
        elif a >= 0.5 * b:
            return self.p[3].eval(locals)
        else:
            t = 0.5 * math.sin(a * math.pi / b)
            return (0.5 - t) * self.p[2].eval(locals) + (0.5 + t) * self.p[3].eval(locals)

    def __str__(self) -> str:
        return f'trans({self.p[0]}, {self.p[1]}, {self.p[2]}, {self.p[3]})'

class ExprUnitSine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.sin(2. * math.pi * self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'sn({self.p[0]})'

class ExprUnitCosine(Expr):
    def eval(self, locals:Dict[str,float]) -> float:
        return math.cos(2. * math.pi * self.p[0].eval(locals))

    def __str__(self) -> str:
        return f'cs({self.p[0]})'


# Each tuple contains: ( operator_name, precedence, argument_count, class )
# Some special cases:
#     argument_count of -2 indicates two arguments, one before and one after the operator.
#     argument_count of -1 indicates a variable number of arguments, but there must be at least one.
#     argument_count of 0 indicates a named constant.
#     negation differs from minus, and has operator precedence between '^' and '*'.
operator_descriptors = [
    ('^', 6, -2, ExprExponent),
    ('/', 5, -2, ExprDivide),
    ('%', 4, -2, ExprModulus),
    ('*', 3, -2, ExprMultiply),
    ('-', 2, -2, ExprSubtract),
    ('+', 1, -2, ExprAdd),
    ('abs', 9, 1, ExprAbsolute),
    ('acos', 9, 1, ExprArcCosine),
    ('acosh', 9, 1, ExprArcCosh),
    ('asin', 9, 1, ExprArcSine),
    ('asinh', 9, 1, ExprArcSinh),
    ('atan', 9, 1, ExprArcTan),
    ('atanh', 9, 1, ExprArcTanh),
    ('ceil', 9, 1, ExprCeil),
    ('clip', 9, 3, ExprClip),
    ('cos', 9, 1, ExprCosine),
    ('cosh', 9, 1, ExprCosh),
    ('cs', 9, 1, ExprUnitCosine),
    ('e', 10, 0, ExprConstant),
    ('erf', 9, 1, ExprErf),
    ('floor', 9, 1, ExprFloor),
	('gamma', 9, 1, ExprGamma),
	('lgamma', 9, 1, ExprLogGamma),
	('log', 9, 1, ExprLog),
	('max', 9, -1, ExprMax),
	('min', 9, -1, ExprMin),
    ('pi', 10, 0, ExprConstant),
	('sign', 9, 1, ExprSign),
	('sin', 9, 1, ExprSine),
	('sinh', 9, 1, ExprSinh),
	('sn', 9, 1, ExprUnitSine),
	('sqrt', 9, 1, ExprSqrt),
	('tan', 9, 1, ExprTan),
	('tanh', 9, 1, ExprTanh),
    ('trans', 9, 4, ExprTrans), # (a, b, x, y) returns x if a < -b/2, y if a > b/2, and interpolates if a is in between
]
OpDescType = Tuple[str,int,int,Any]

named_constants = {
    'pi': math.pi,
    'e': math.e,
}

class OpTrie():
    def __init__(self) -> None:
        self.children:Dict[str,OpTrie] = {}
        self.result:Optional[OpDescType] = None

    def longest_match(self, s:str, start:int=0, sofar:Optional[OpDescType]=None) -> Optional[OpDescType]:
        if self.result is not None:
            sofar = self.result
        if len(s) > start and s[start] in self.children:
            return self.children[s[start]].longest_match(s, start + 1, sofar)
        return sofar

def build_op_trie(op_descrs:List[OpDescType], pos:int) -> OpTrie:
    node = OpTrie()
    while len(op_descrs) > 0:
        beflen = len(op_descrs)
        c = ''
        clust:List[OpDescType] = []
        for i in reversed(range(len(op_descrs))):
            if len(op_descrs[i][0]) <= pos:
                assert node.result is None, 'Internal error: multiple matches in the trie'
                op_descrs[i], op_descrs[-1] = op_descrs[-1], op_descrs[i] # swap to the end
                node.result = op_descrs.pop()
            else:
                if len(c) == 0:
                    c = op_descrs[i][0][pos]
                if op_descrs[i][0][pos] == c:
                    op_descrs[i], op_descrs[-1] = op_descrs[-1], op_descrs[i] # swap to the end
                    clust.append(op_descrs.pop())
        if len(c) > 0:
            node.children[c] = build_op_trie(clust, pos + 1)
        assert len(op_descrs) < beflen, 'Internal error: failed to reduce the list'
    return node

op_trie = build_op_trie(operator_descriptors, 0)


# Consumes an expression. Returns the location and descriptor of the lowest-precedence operator.
def find_operator_with_lowest_precedence(tok:tokenizer.Tokenizer) -> Tuple[int, Optional[OpDescType]]:
    op_precedence = 1000000
    op_desc:Optional[OpDescType] = None
    op_pos:int = -1
    prev_cc = 4
    paren_nests = 0
    s = tok.peek()
    for i in range(len(s)):
        # Track parentheses
        if s[i] == '(':
            paren_nests += 1
        elif s[i] == ')':
            paren_nests -= 1
            if paren_nests < 0:
                tok.advance(i)
                tok.err(f'")" without matching "("')

        # Look for an operator here
        cc = char_class(s[i])
        if paren_nests == 0 and cc != prev_cc:
            op_descriptor = op_trie.longest_match(s, i)
            if op_descriptor is not None:
                after_op_pos = i + len(op_descriptor[0])
                if after_op_pos >= len(s) or char_class(s[after_op_pos]) != cc:
                    # Found an operator. See if it has lower precedence
                    if op_descriptor[1] < op_precedence:
                        op_precedence = op_descriptor[1]
                        op_desc = op_descriptor
                        op_pos = i
        prev_cc = cc
    return op_pos, op_desc

# Parses the parentheses-bounded comma-separated arguments to a function call, ala "(expr1, expr2, expr3)".
def parse_args(tok:tokenizer.Tokenizer, op_name:str) -> List[Expr]:
    s = tok.peek()
    if s[0] != '(':
        tok.err(f'Expected a "(" after "{op_name}"')
    if s[-1] != ')':
        tok.err(f'"(" without matching ")" following "{op_name}"')
    args:List[Expr] = []
    nests = 1
    while tok.remaining() > 0:
        tok.advance(1)
        s = tok.peek()
        for i in range(len(s)):
            if s[i] == '(':
                nests += 1
            elif s[i] == ')':
                nests -= 1
                if nests == 0:
                    if i + 1 != len(s):
                        tok.advance(i + 1)
                        tok.err(f'Unexpected symbols following arguments for "{op_name}"')
                    left_expr, _ = tok.split(i)
                    args.append(parse_expr(left_expr))
                    return args
            elif nests == 1 and s[i] == ',':
                left_expr, tok = tok.split(i)
                args.append(parse_expr(left_expr))
                break
    tok.err(f'Imbalanced parenthesis after "{op_name}"')
    return [] # This line is never reached

# s is assumed to already be stripped of whitespace on both ends
def parse_expr(tok:tokenizer.Tokenizer) -> Expr:
    tok.rtrim()
    tok.skip_ws()
    op_pos, op_desc = find_operator_with_lowest_precedence(tok)
    if op_desc is not None:
        if op_desc[2] == -2: # math operator
            left_tok, right_tok = tok.split(op_pos)
            left_tok.rtrim()
            right_tok.advance(1) # drop the operator
            if op_desc[0] == '-' and left_tok.remaining() == 0:
                left_expr:Expr = ExprConstant(0.)
            else:
                left_expr = parse_expr(left_tok)
            right_expr = parse_expr(right_tok)
            combined:Expr = op_desc[3]([left_expr, right_expr])
            if isinstance(left_expr, ExprConstant) and isinstance(right_expr, ExprConstant):
                return ExprConstant(combined.eval({})) # reduce the expression (because we can)
            else:
                return combined
        elif op_desc[2] == 0: # named constants
            if tok.remaining() != len(op_desc[0]):
                tok.err(f'Superfluous characters in "{tok.peek()}"')
            if not op_desc[0] in named_constants:
                tok.err(f'Internal error. Unrecognized named constant: "{op_desc[0]}"')
            return ExprConstant(named_constants[op_desc[0]])
        else: # named operator
            tok.advance(op_pos+len(op_desc[0]))
            tok.skip_ws()
            args = parse_args(tok, op_desc[0])
            if op_desc[2] > 0 and len(args) != op_desc[2]:
                tok.err(f'{op_desc[0]} expects {op_desc[2]} args. {len(args)} {"was" if len(args) == 1 else "were"} given')
            elif op_desc[2]== -1 and len(args) < 1:
                tok.err(f'{op_desc[0]} expects at least 1 arg. None were given')
            combined = op_desc[3](args)
            if all([isinstance(x, ExprConstant) for x in args]):
                return ExprConstant(combined.eval({})) # reduce the expression (because we can)
            else:
                return combined
    else:
        # Check for empty string
        if tok.remaining() == 0:
            tok.err(f'Operator with unexpected lack of arguments')

        # Look for bounding parens
        s = tok.peek()
        if s[0] == '(':
            if s[-1] == ')':
                tok.shorten(1)
                tok.advance(1)
                return parse_expr(tok)
            else:
                tok.err(f'"(" without matching ")" in "{s}"')

        # Try interpretting as a numeric value
        try:
            return ExprConstant(float(s))
        except:
            pass

        # It must be a variable
        return ExprVariable(s)

def parse_expr_str(s:str, line_num:int) -> Expr:
    tok = tokenizer.Tokenizer(s, [(0, line_num)])
    return parse_expr(tok)
