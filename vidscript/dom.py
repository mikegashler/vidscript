from typing import Union, List, Callable, Dict, Optional, Tuple, Set
import os
import sys
from . import expr, renderer

Arg = Tuple[str,str,int]
FlexArg = Union[Arg, Tuple[str,str]]

def flexArgToArg(fa:FlexArg) -> Arg:
    return (fa[0], fa[1], fa[2] if len(fa) > 2 else 0) # type: ignore

# Returns the line number of the caller
def this_line() -> int:
    return int(sys._getframe(1).f_lineno)

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

# Counts the number of whitespace characters at the beginning of the line.
# Tabs are counted as four spaces.
def count_leading_spaces(s:str, line_num:int) -> int:
    space_count = 0
    for i in range(len(s)):
        if s[i] > ' ':
            return space_count
        elif s[i] == ' ':
            space_count += 1
        elif s[i] == '\t':
            space_count += 4
        else:
            raise ValueError(f'Error on line {line_num}: Only spaces and tabs are allowed as leading characters. Got ascii char {ord(s[i])}.')
    raise ValueError(f'Error on line {line_num}: Internal error. Only whitespace lines should not reach here.')

# Checks for a '.' operator, suggesting special handling is needed.
# Otherwise, passes off to expr.parse_expr_str to parse the expression.
def parse_expr(arg:Arg, all_blocks:Optional[Dict[str,renderer.Block]]=None, enclosing_block:Optional[renderer.Block]=None, resolved_generics:Optional[Dict[str,renderer.Block]]=None) -> expr.Expr:
    generics:Dict[str, str] = enclosing_block.generics if enclosing_block else {}
    last_dot = arg[0].rfind('.')
    if arg[0] == 'text' or arg[0] == 'filename':
        return expr.ExprStr(arg[1])
    elif last_dot < 0 and arg[0] in generics:
        if resolved_generics is not None and all_blocks is not None:
            resolved_generics[arg[0]] = renderer.find_block(arg[1], all_blocks, arg[2])
        return expr.ExprStr(arg[1])
    elif last_dot >= 0 and resolved_generics is not None:
        if arg[0][:last_dot] in resolved_generics:
            inner_enclosing_block = resolved_generics[arg[0][:last_dot]]
            inner_generics = inner_enclosing_block.generics
            if arg[0][last_dot+1:] in inner_generics:
                assert all_blocks is not None, f'Error on line {arg[2]}: Internal error'
                resolved_generics[arg[0]] = renderer.find_block(arg[1], all_blocks, arg[2])
                return expr.ExprStr(arg[1])
            else:
                return expr.parse_expr_str(arg[1], arg[2])
        else:
            raise ValueError(f'Error on line {arg[2]}: Generic type not yet set: {arg[0][:last_dot]}')
    else:
        return expr.parse_expr_str(arg[1], arg[2])

def parse_local_var(arg:Arg, all_blocks:Dict[str,renderer.Block], block:renderer.Block, locals:Dict[str,expr.Expr]) -> renderer.Part:
    if arg[0] in renderer.affine_transforms or arg[0] in renderer.color_transforms:
        raise ValueError(f'Error on line {arg[2]}: local variable {arg[0]} conflicts with built-in parameter. (Either pick a unique name, or put this under some part.)')
    ex = parse_expr(arg)
    ex.check_variables(arg[2], block.params, locals)
    locals[arg[0]] = ex
    return renderer.Part(all_blocks, '', [], None, None, [(arg[0], ex)], [])

def check_arg(blocks:List[renderer.Block], name:str, line_num:int) -> None:
    if name in renderer.affine_transforms:
        return
    if name in renderer.color_transforms:
        return
    for block in blocks:
        if name in block.params:
            continue
        if name in block.generics:
            continue
        raise ValueError(f'Error on line {line_num}: {block.name} has no parameter {name}. (If you are trying to declare a new variable, it should be directly within the type, not within a part within the type.)')

class Part():
    def __init__(self, name:str, arg_list:Optional[List[FlexArg]]=None, line_num:int=0) -> None:
        assert len(name) > 0, f'Error on line {line_num}: A Part requires a name'
        for c in name:
            if is_whitespace(c):
                raise ValueError(f'Error on line {line_num}: names may not contain whitespace')
        self.name = name
        self.arg_list = [] if arg_list is None else [ flexArgToArg(fa) for fa in arg_list ]
        self.line_num = line_num

    def add_arg(self, arg:Arg) -> None:
        self.arg_list.append(arg)

    def __str__(self) -> str:
        return f'    {self.name}\n' + ''.join([f'        {arg[0]} = {arg[1]}\n' for arg in self.arg_list])

    def __repr__(self) -> str:
        return f'Part(name={self.name},line={self.line_num},args={repr(self.arg_list)})'

    def to_block_part(
        self,
        all_blocks:Dict[str,renderer.Block],
        block:renderer.Block,
        locals:Dict[str,expr.Expr],
        generic_bindings:Dict[str,List[renderer.Block]],
    ) -> renderer.Part:
        # Make sure type exists
        if self.name in generic_bindings:
            part_blocks = generic_bindings[self.name]
        else:
            part_blocks = [renderer.find_block(self.name, all_blocks, self.line_num)]

        # Sort the args into appropriate lists
        expr_beg:Optional[expr.Expr] = None
        expr_end:Optional[expr.Expr] = None
        args_pre:List[Tuple[str,expr.Expr]] = []
        args_post:List[Tuple[str,expr.Expr]] = []
        resolved_generics:Dict[str,renderer.Block] = {}
        for arg in self.arg_list:
            ex = parse_expr(arg, all_blocks, part_blocks[0] if len(part_blocks) == 1 else None, resolved_generics)
            line_num = arg[2]
            ex.check_variables(line_num, block.params, locals)
            dotspot = arg[0].rfind('.')
            if arg[0] == 'beg':
                expr_beg = ex
            elif arg[0] == 'end':
                expr_end = ex
            elif arg[0] in renderer.color_transforms:
                # Got an argument to process on the way out
                args_post.append((arg[0], ex))
            else:
                # Got a regular argument
                if dotspot >= 0:
                    # Check generics
                    front = arg[0][:dotspot]
                    back = arg[0][dotspot+1:]
                    if not front in resolved_generics:
                        raise ValueError(f'Error on line {self.line_num}: Generic type not yet set: "{front}"')
                    dest_block = resolved_generics[front]
                    check_arg([dest_block], back, arg[2])
                else:
                    check_arg(part_blocks, arg[0], arg[2])
                args_pre.append((arg[0], ex))
        return renderer.Part(all_blocks, self.name, part_blocks, expr_beg, expr_end, args_pre, args_post)



class Type():
    def __init__(self, name:str, param_list:Optional[List[FlexArg]]=None, part_list:Optional[List[Union[Part,FlexArg]]]=None, line_num:int=0) -> None:
        assert len(name) > 0, f'Error on line {line_num}: A Type requires a name'
        for c in name:
            if is_whitespace(c):
                raise ValueError(f'Error on line {line_num}: names may not contain whitespace')
        self.name = name
        self.param_list:List[Arg] = [] if param_list is None else [ flexArgToArg(fa) for fa in param_list ]
        self.part_list:List[Union[Part,Arg]] = [] if part_list is None else [ p if isinstance(p, Part) else flexArgToArg(p) for p in part_list ]
        self.line_num = line_num

    def add_param(self, param:Arg) -> None:
        self.param_list.append(param)

    def add_part(self, part:Part) -> None:
        self.part_list.append(part)

    def add_local(self, local:Arg) -> None:
        self.part_list.append(local)

    def get_part(self, name:str, index:int) -> Part:
        i = 0
        for p in self.part_list:
            if isinstance(p, Part) and p.name == name:
                if i == index:
                    return p
                else:
                    i += 1
        raise ValueError(f'{self.name} only has {i} parts of type {name}')

    def get_local(self, name:str) -> Arg:
        for l in self.part_list:
            if isinstance(l, tuple) and l[0] == name:
                return l
        raise ValueError(f'{self.name} has no local variable named {name}')

    def __str__(self) -> str:
        return f'{self.name} ' + ' '.join([f'{param[0]}={param[1]}' for param in self.param_list]) + '\n' + ''.join([(f'{str(part)}' if isinstance(part, Part) else f'    {part[0]} = {part[1]}\n') for part in self.part_list])

    def __repr__(self) -> str:
        return f'Type(name={self.name}, line={self.line_num}, params={repr(self.param_list)})'

    def to_block(self) -> renderer.Block:
        block = renderer.Block(self.name, self.line_num)

        # Load the params
        for param in self.param_list:
            ex = parse_expr(param)
            if isinstance(ex, expr.ExprVariable):
                block.generics[param[0]] = ex.name
            else:
                block.params[param[0]] = ex.eval({})

        # Add some default params
        if not 'x' in block.params:
            block.params['x'] = 0.
        if not 'y' in block.params:
            block.params['y'] = 0.
        if not 'z' in block.params:
            block.params['z'] = 0.
        if not 't' in block.params:
            block.params['t'] = 0.
        if not 'beg' in block.params:
            block.params['beg'] = 0.
        if not 'end' in block.params:
            block.params['end'] = 1.
        return block

def _find_all_generic_bindings(typ:Type, generic_name:str, default_type:str, type_list:List[Type], all_blocks:Dict[str,renderer.Block]) -> List[renderer.Block]:
    def_block = renderer.find_block(default_type, all_blocks, typ.line_num)
    bindings:List[renderer.Block] = [def_block]
    bindings_set:Set[str] = set()
    bindings_set.add(default_type)
    for inner_typ in type_list:
        for inner_part in inner_typ.part_list:
            if isinstance(inner_part, Part) and inner_part.name == typ.name:
                for inner_arg in inner_part.arg_list:
                    if inner_arg[0] == generic_name and not inner_arg[1] in bindings_set:
                        inner_block = renderer.find_block(inner_arg[1], all_blocks, inner_part.line_num)
                        bindings_set.add(inner_arg[1])
                        bindings.append(inner_block)
    return bindings

class Script():
    def __init__(self, type_list:Optional[List[Type]]=None) -> None:
        if type_list is None:
            type_list = []
        self.type_list = type_list

    def add_type(self, typ:Type) -> None:
        self.type_list.append(typ)

    def get_type(self, name:str) -> Type:
        for t in self.type_list:
            if t.name == name:
                return t
        raise ValueError(f'{name} not found')

    def __str__(self) -> str:
        return '\n'.join([str(typ) for typ in self.type_list])

    # This method does the second phase of parsing.
    # It links all the parts together to prepare for rendering.
    def to_blocks(self) -> Dict[str,renderer.Block]:
        # Create a block for each type
        all_blocks:Dict[str,renderer.Block] = {}
        for typ in self.type_list:
            all_blocks[typ.name] = typ.to_block()

        # Instantiate the bodies
        for typ in self.type_list:
            block = all_blocks[typ.name]

            # Make a list of all generic instantiations
            generic_bindings:Dict[str,List[renderer.Block]] = {}
            for generic_name in block.generics:
                generic_bindings[generic_name] = _find_all_generic_bindings(typ, generic_name, block.generics[generic_name], self.type_list, all_blocks)
            locals:Dict[str,expr.Expr] = {}
            for part in typ.part_list:
                if isinstance(part, Part):
                    block_part = part.to_block_part(all_blocks, block, locals, generic_bindings)
                else:
                    block_part = parse_local_var(part, all_blocks, block, locals)
                block.parts.append(block_part)

        return all_blocks

def parse_arg(lines:List[str], line_nums:List[int]) -> Arg:
    line_num = line_nums[0] if len(line_nums) > 0 else 0
    s = ''.join(lines)
    eq = s.find('=')
    if eq < 0:
        raise ValueError(f'Error on line {line_num}: Expected an "="')
    name = ltrim(rtrim(s[:eq]))
    exp = ltrim(rtrim(s[eq+1:]))
    return name, exp, line_num

def parse_args(lines:List[str], line_nums:List[int]) -> List[FlexArg]:
    # Test for a line with multiple args on it
    multi_arg_line = False
    if len(lines) == 1:
        eq = lines[0].find('=')
        if eq < 0:
            return []
        eq2 = lines[0].find('=', eq + 1)
        if eq2 >= 0:
            multi_arg_line = True

    # Parse all the args
    args:List[FlexArg] = []
    if multi_arg_line:
        # Multiple args on one line
        head_col = 0
        while True:
            # Find where the arg begins and ends
            eq = lines[0].find('=', head_col)
            if eq < 0:
                break
            eq2 = lines[0].find('=', eq + 1)
            next_start = len(lines[0])
            if eq2 >= 0:
                # Absorb the name and whitespace that precedes the second '='
                tail_col = eq2 - 1
                while tail_col > 0 and is_whitespace(lines[0][tail_col - 1]):
                    tail_col -= 1
                while tail_col > 0 and not is_whitespace(lines[0][tail_col - 1]):
                    tail_col -= 1
                next_start = tail_col
                while tail_col > 0 and is_whitespace(lines[0][tail_col - 1]):
                    tail_col -= 1
            else:
                tail_col = len(lines[0])

            # Parse it and move on to the next one
            args.append(parse_arg([lines[0][head_col:tail_col]], [line_nums[0] if len(line_nums) > 0 else 0]))
            head_col = next_start
    else:
        # One arg per line
        head_line = 0
        while True:
            # Find the next '='
            if head_line >= len(lines):
                break
            eq = lines[head_line].find('=')
            assert eq >= 0, f'Error on line {line_nums[head_line] if len(line_nums) > head_line else 1}: Expected an "="'

            # Find the tail of the expression
            tail_line = head_line + 1
            while tail_line < len(lines) and lines[tail_line].find('=') < 0:
                tail_line += 1

            # Parse it and move on to the next one
            args.append(parse_arg(lines[head_line:tail_line], line_nums[head_line:tail_line]))
            head_line = tail_line
    return args

def parse_part(lines:List[str], line_nums:List[int]) -> Part:
    name = ltrim(rtrim(lines[0]))
    arg_list = parse_args(lines[1:], line_nums[1:])
    return Part(name, arg_list, line_nums[0] if len(line_nums) > 0 else 0)

def parse_parts(lines:List[str], line_nums:List[int]) -> List[Union[Part, FlexArg]]:
    if len(lines) < 1:
        return []
    part_indents = count_leading_spaces(lines[0], line_nums[0])
    tail = 0
    parts:List[Union[Part,FlexArg]] = []
    while True:
        # Find the head and tail of the part or variable
        head = tail
        if head >= len(lines):
            break
        tail = head + 1
        while tail < len(lines) and count_leading_spaces(lines[tail], line_nums[tail]) > part_indents:
            tail += 1

        # Determine whether it is a part or variable
        s = lines[head][part_indents:]
        ws = find_whitespace(s, part_indents)
        if ws < 0:
            ws = len(s)
        eq = s.find('=')
        if eq < 0:
            eq = len(s)
        s = ltrim(s[min(ws,eq):])
        if len(s) > 0 and s[0] == '=':
            # Local variable
            parts.append(parse_arg(lines[head:tail], line_nums[head:tail]))
        else:
            # Part
            parts.append(parse_part(lines[head:tail], line_nums[head:tail]))
    return parts

def parse_type(lines:List[str], line_nums:List[int]) -> Type:
    # Parse the name
    s = lines[0]
    ws = find_whitespace(s)
    name_end = ws if ws >= 0 else len(s)
    name = s[:name_end]
    if name in renderer.built_in_blocks:
        raise ValueError(f'Error on line {line_nums[0]}: {name} conflicts with a built-in object')
    s = ltrim(s[name_end+1:])

    # Parse the params and parts
    param_list = parse_args([s], [line_nums[0]])
    part_list = parse_parts(lines[1:], line_nums[1:])
    return Type(name, param_list, part_list, line_nums[0] if len(line_nums) > 0 else 0)

def parse_script(lines:List[str], start_line:int=0) -> Script:
    v = Script()
    type_lines:List[str] = []
    type_line_nums:List[int] = []
    block_comment = False
    for line_num, s in enumerate(lines):
        # Absorb comments and blank lines
        if block_comment:
            if s.find('<#') >= 0:
                block_comment = False
            continue
        comment_index = s.find('#')
        if comment_index >= 0:
            if comment_index + 1 < len(s) and s[comment_index + 1] == '>':
                block_comment = True
            s = s[:comment_index]
        s = rtrim(s)
        if len(s) == 0:
            continue

        # Gather type lines
        if (s[0] > ' '):
            # Process the previous type
            if len(type_lines) > 0:
                v.add_type(parse_type(type_lines, type_line_nums))
            type_lines = []
            type_line_nums = []
        type_lines.append(s)
        type_line_nums.append(line_num + start_line)

    # Process the previous type
    if len(type_lines) > 0:
        v.add_type(parse_type(type_lines, type_line_nums))
    return v


if __name__ == "__main__":
	print('yo')
