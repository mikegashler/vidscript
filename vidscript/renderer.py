from typing import List, Tuple, Dict, Optional, Union, Any, cast, Set
import math
from PIL import Image
from . import expr

invisibility_cloak = 2.0 # This special alpha value indicates that no parts behind it in the same type should be rendered
image_cache:Dict[str,Image.Image] = {}

def load_image(filename:str) -> Image.Image:
    if filename in image_cache:
        return image_cache[filename]
    im = Image.open(filename)
    image_cache[filename] = im
    return im


TOO_FAR = 1e38
ParamType = Union[float,str,Dict[str,Any]]

def translate_x(xyzt:List[float], val:float) -> None:
    xyzt[0] -= val

def translate_y(xyzt:List[float], val:float) -> None:
    xyzt[1] -= val

def translate_z(xyzt:List[float], val:float) -> None:
    xyzt[2] -= val

def set_t(xyzt:List[float], val:float) -> None:
    xyzt[3] = val

def scale(xyzt:List[float], val:float) -> None:
    if val == 0.:
        xyzt[0] = 100000000.
    else:
        xyzt[0] /= val
        xyzt[1] /= val

def hscale(xyzt:List[float], val:float) -> None:
    if val == 0.:
        xyzt[0] = 100000000.
    else:
        xyzt[0] /= val

def vscale(xyzt:List[float], val:float) -> None:
    if val == 0.:
        xyzt[1] = 100000000.
    else:
        xyzt[1] /= val

def hshear(xyzt:List[float], val:float) -> None:
    xyzt[0] -= xyzt[1] * val

def vshear(xyzt:List[float], val:float) -> None:
    xyzt[1] -= xyzt[0] * val

def rotate(xyzt:List[float], val:float) -> None:
    val *= (-2. * math.pi)
    s = math.sin(val)
    c = math.cos(val)
    xyzt[0], xyzt[1] = xyzt[0] * c - xyzt[1] * s, xyzt[0] * s + xyzt[1] * c


affine_transforms = {
    'x': translate_x,
    'y': translate_y,
    'z': translate_z,
    't': set_t,
    'scale': scale,
    'hscale': hscale,
    'vscale': vscale,
    'hshear': hshear,
    'vshear': vshear,
    'rotate': rotate,
}

def set_red(drgba:List[float], val:float) -> None:
    drgba[1] = val

def set_green(drgba:List[float], val:float) -> None:
    drgba[2] = val

def set_blue(drgba:List[float], val:float) -> None:
    drgba[3] = val

def set_opacity(drgba:List[float], val:float) -> None:
    drgba[4] *= val

def set_cloak(drgba:List[float], val:float) -> None:
    if val < 0:
        drgba[4] = invisibility_cloak

color_transforms = {
    'red': set_red,
    'green': set_green,
    'blue': set_blue,
    'opacity': set_opacity,
    'cloak': set_cloak,
}

def ind_print(indent: int, message: str) -> None:
    for i in range(indent):
        print('  ', end='')
    print(message)

def print_locals(indent: int, locals:Dict[str,ParamType]) -> None:
    for k in locals:
        ind_print(indent, f'[{k}: {locals[k]}]')

# Represents a part in a block
class Part():
    # flavor: 0=local variable, 1=strongly typed, 2=generic typed
    def __init__(self, all_blocks:Dict[str,'Block'], type_name:str, flavor:int, beg_expr:Optional[expr.Expr], end_expr:Optional[expr.Expr], args_pre:List[Tuple[str,expr.Expr]], args_post:List[Tuple[str,expr.Expr]]) -> None:
        # Make sure it's either a regular part or a local variable
        assert (len(type_name) > 0) or (len(type_name) == 0 and flavor == 0 and len(args_pre) == 1 and len(args_post) == 0)
        self.type_name = type_name
        self.flavor = flavor
        self.block: Optional['Block'] = find_block(type_name, all_blocks, 0) if flavor == 1 else None
        self.beg_expr = beg_expr
        self.end_expr = end_expr
        self.args_pre = args_pre
        self.args_post = args_post
        if self.block:
            self.block.init_part(self) # Do any necessary special type-specific initialization (for example, load images, render text, etc.)

    # depth is for debugging only. When it is >= 0, debug output will be produced.
    def render_pixel(self, all_blocks:Dict[str,'Block'], locals:Dict[str,ParamType], generics:Dict[str,str], depth:int) -> List[float]:
        if depth >= 0 and len(self.type_name) > 0:
            ind_print(2 * depth + 1, f'part {self.type_name}')
        if self.flavor == 0:
            # Local variable
            name, expr = self.args_pre[0]
            locals[name] = expr.eval(cast(Dict[str,float], locals))
            return [TOO_FAR, 0., 0., 0., 0.]
        elif self.block: # flavor == 1:
            # strongly typed part
            block:Block = self.block
        else:
            # generic typed part
            type_name:str = cast(str, locals[self.type_name])
            _block = find_block(type_name, all_blocks, 0)
            assert _block, f'Internal error: Block not found: {type_name}'
            block = _block

        # Compute the inner_locals for the callee
        inner_locals = block.params.copy()
        for p in block.generics:
            inner_locals[p] = block.generics[p]
        xyzt = [ cast(float, locals['x']), cast(float, locals['y']), cast(float, locals['z']), cast(float, locals['t']) ]
        for argname, argexpr in reversed(self.args_pre):
            if depth >= 0:
                ind_print(2 * depth + 2, f'{argname} = {str(argexpr)}')
            if argname in affine_transforms:
                affine_transforms[argname](xyzt, argexpr.eval(cast(Dict[str,float], locals)))
            else:
                val = argexpr.eval(cast(Dict[str,float], locals))
                inner_locals[argname] = val
        inner_locals['x'], inner_locals['y'], inner_locals['z'], inner_locals['t'] = xyzt
        inner_locals.update(generics)
        inner_locals['ft'] = locals['ft']

        # Render the block
        drgba = block.render_pixel(self, all_blocks, inner_locals, depth + 1)

        # Post-process
        for argname, argexpr in self.args_post:
            color_transforms[argname](drgba, argexpr.eval(cast(Dict[str,float], locals)))

        # print(f'Returning drgba={drgba}')
        if depth >= 0 and drgba[0] < TOO_FAR:
            ind_print(2 * depth + 2, f'dist={drgba[0]} red={drgba[1]} green={drgba[2]} blue={drgba[3]} opacity={drgba[4]}')
        return drgba


# Combines semi-transparent pixels overlapping in z-order
def combine_pixels(pixels:List[List[float]]) -> List[float]:
    pixels.sort(key=lambda x: x[0]) # smaller distance means closer to the camera, and should be on top
    # print(f'combine_pixels: pixels={pixels}')
    dist = pixels[0][0]
    r = g = b = a = 0.
    for pix in pixels:
        if pix[4] == invisibility_cloak:
            if dist < pix[0]:
                break
            else:
                return [TOO_FAR, r, g, b, a]
        r = (a * r + (1. - a) * pix[1])
        g = (a * g + (1. - a) * pix[2])
        b = (a * b + (1. - a) * pix[3])
        a = min(1., a + ((1. - a) * pix[4]))
        if a >= 0.97:
            a = 1. # For efficiency, let's just snap to fully opaque. No one can see such negligible transparency anyway.
            break
    return [dist, r, g, b, a]


# Represents a parsed block of animation code
class Block():
    def __init__(self, name:str, line_num:int, body_lines:List[List[Tuple[str,int]]] = []) -> None:
        self.name = name
        self.line_num = line_num
        self.params:Dict[str,ParamType] = {}
        self.generics:Dict[str,str] = {} # Maps from name to default value
        self.body_lines = body_lines # temporary. Used only while parsing
        self.parts:List[Part] = []

    def init_part(self, part:Part) -> None:
        pass

    # Returns a distance, red, green, blue, opacity for the specified pixel in the specified frame.
    # (red, green, and blue range from 0-256. Opacity is from 0-1.)
    # depth is for debugging only. When it is >= 0, debug output will be produced.
    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,"Block"], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)

        # Separate out the names containing a dot
        generics:Dict[str, Dict[str,str]] = {}
        condemned:Set[str] = set()
        for name in locals:
            dot_spot = name.find('.')
            if dot_spot >= 0:
                front = name[:dot_spot]
                back = name[dot_spot+1:]
                if front in generics:
                    generics[front][back] = cast(str, locals[name])
                else:
                    generics[front] = { back: cast(str, locals[name]) }
                condemned.add(name)
        if len(condemned) > 0:
            locals = { l:locals[l] for l in locals if not l in condemned }

        pixels:List[List[float]] = []
        t = cast(float, locals['t'])
        ft = cast(float, locals['ft'])
        for part in self.parts:
            beg_val = 0.
            end_val = 1.
            if part.beg_expr is not None or part.end_expr is not None:
                if part.beg_expr is not None:
                    beg_val = part.beg_expr.eval(cast(Dict[str,float], locals))
                if part.end_expr is not None:
                    end_val = part.end_expr.eval(cast(Dict[str,float], locals))
                locals['ft'] = ft / max(1e-6, end_val - beg_val)
            if beg_val <= t < end_val:
                locals['t'] = (t - beg_val) / max(1e-6, end_val - beg_val)
                pix = part.render_pixel(blocks, locals, generics[part.type_name] if part.type_name in generics else {}, depth)
                if pix[0] < TOO_FAR:
                    pixels.append(pix)
            locals['ft'] = ft

        # Restore the original values so locals can be reused for the next pixel
        locals['t'] = t
        if len(pixels) == 0:
            return [TOO_FAR, 0., 0., 0., 0.]
        elif len(pixels) == 1:
            if pixels[0][4] == invisibility_cloak:
                return [TOO_FAR, 0., 0., 0., 0.]
            else:
                return pixels[0]
        else:
            pix = combine_pixels(pixels)
            if depth >= 0:
                ind_print(2 * depth, f'combined {len(pixels)} pixels to dist={pix[0]} red={pix[1]} green={pix[2]} blue={pix[3]} opacity={pix[4]}')
            return pix

class BlockCircle(Block):
    def __init__(self) -> None:
        super().__init__('circle', -1, [])
        self.params['from'] = 0.
        self.params['to'] = 1.
        self.params['red'] = 255.
        self.params['green'] = 128.
        self.params['blue'] = 128.
        self.params['thickness'] = 50.
        self.params['opacity'] = 1.

    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,Block], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)
        x = cast(float, locals['x'])
        y = cast(float, locals['y'])
        z = cast(float, locals['z'])
        thickness = cast(float, locals['thickness'])
        d = x * x + y * y
        if (50. - thickness) * (50. - thickness) <= d < 2500.:
            _from = cast(float, locals['from'])
            _to = cast(float, locals['to'])
            if math.fabs(_to - _from) >= 1.:
                return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity'])]
            if _to < _from:
                _from, _to = _to, _from
            adj = -math.floor(_from + 0.5)
            _from += adj
            _to += adj
            theta = (math.atan2(y, x) / (2.0 * math.pi))
            if _from < theta <= _to:
                if depth >= 0:
                    ind_print(2 * depth, f'hit!')
                return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity'])]
            elif _to > 0.5 and theta < _to - 1.:
                if depth >= 0:
                    ind_print(2 * depth, f'hit!')
                return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity'])]
        if depth >= 0:
            ind_print(2 * depth, f'miss')
        return [TOO_FAR, 0., 0., 0., 0.]

class BlockSquare(Block):
    def __init__(self) -> None:
        super().__init__('square', -1, [])
        self.params['red'] = 128.
        self.params['green'] = 255.
        self.params['blue'] = 128.
        self.params['opacity'] = 1.

    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,Block], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)
        x = cast(float, locals['x'])
        y = cast(float, locals['y'])
        z = cast(float, locals['z'])
        if x >= -50. and x < 50. and y >= -50. and y < 50.:
            if depth >= 0:
                ind_print(2 * depth, f'hit!')
            return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity'])]
        else:
            if depth >= 0:
                ind_print(2 * depth, f'miss')
            return [TOO_FAR, 0., 0., 0., 0.]

def side(x:float, y:float, x2:float, y2:float, x3:float, y3:float) -> float:
    return (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)

class BlockTriangle(Block):
    def __init__(self) -> None:
        super().__init__('triangle', -1, [])
        self.params['x1'] = 0.
        self.params['y1'] = 50.
        self.params['x2'] = -75. / math.sqrt(3)
        self.params['y2'] = -25
        self.params['x3'] = 75. / math.sqrt(3)
        self.params['y3'] = -25
        self.params['red'] = 128.
        self.params['green'] = 128.
        self.params['blue'] = 255.
        self.params['opacity'] = 1.

    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,Block], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)
        x = cast(float, locals['x'])
        y = cast(float, locals['y'])
        z = cast(float, locals['z'])
        x1 = cast(float, locals['x1'])
        y1 = cast(float, locals['y1'])
        x2 = cast(float, locals['x2'])
        y2 = cast(float, locals['y2'])
        x3 = cast(float, locals['x3'])
        y3 = cast(float, locals['y3'])
        a = side(x, y, x1, y1, x2, y2)
        b = side(x, y, x2, y2, x3, y3)
        c = side(x, y, x3, y3, x1, y1)
        side1 = (a < 0.) and (b < 0.) and (c < 0.)
        side2 = (a > 0.) and (b > 0.) and (c > 0.)
        if (side1 or side2):
            if depth >= 0:
                ind_print(2 * depth, f'hit!')
            return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity'])]
        else:
            if depth >= 0:
                ind_print(2 * depth, f'miss')
            return [TOO_FAR, 0., 0., 0., 0.]

class BlockImage(Block):
    def __init__(self) -> None:
        super().__init__('image', -1, [])
        self.params['filename'] = 0.

    def init_part(self, part:Part) -> None:
        # Find the filename in args_pre, and remove it
        filename = ''
        for i, tup in reversed(list(enumerate(part.args_pre))):
            arg_name, arg_expr = tup
            if arg_name == 'filename':
                if not isinstance(arg_expr, expr.ExprStr):
                    raise ValueError('Internal error: Expected an ExprStr')
                filename = arg_expr.name
                del part.args_pre[i]

        # Load the image and attach it to the part
        im = load_image(filename)
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        part.image = im # type: ignore

    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,Block], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)
        x = cast(float, locals['x'])
        y = cast(float, locals['y'])
        z = cast(float, locals['z'])
        image = outer_part.image # type: ignore
        wid, hgt = image.size
        half_wid, half_hgt = wid // 2, hgt // 2
        if x < -half_wid or y < -half_hgt or x >= image.size[0] - half_wid or y >= image.size[1] - half_hgt:
            if depth >= 0:
                ind_print(2 * depth, f'miss')
            return [TOO_FAR, 0., 0., 0., 0.]
        else:
            r, g, b, a = image.getpixel((int(x) + half_wid, image.size[1] - 1 - (int(y) + half_hgt)))
            if depth >= 0:
                ind_print(2 * depth, f'hit!')
            return [z, float(r), float(g), float(b), a / 256.]

font_starts = [
    0,30,48,75,112,147,206,252,267,287,308,335,373,389,412,426,
    446,484,517,552,588,623,659,695,730,766,807,828,845,883,919,958,
    995,1058,1107,1151,1199,1246,1288,1325,1376,1422,1438,1476,1523,1561,1615,1659,
    1711,1752,1804,1847,1889,1932,1975,2017,2078,2121,2163,2205,2222,2242,2265,2297,
    2337,2357,2394,2432,2467,2506,2541,2564,2605,2644,2658,2679,2715,2732,2789,2827,
    2867,2904,2946,2968,3003,3027,3064,3099,3149,3184,3220,3253,3282,3298,3323,3356,
    3387
]

class BlockLabel(Block):
    def __init__(self) -> None:
        super().__init__('label', -1, [])
        self.params['text'] = 0.
        self.params['red'] = 128.
        self.params['green'] = 255.
        self.params['blue'] = 255.
        self.params['opacity'] = 1.

    def init_part(self, part:Part) -> None:
        # Find the text in args_pre, and remove it
        text = ''
        for i, tup in reversed(list(enumerate(part.args_pre))):
            arg_name, arg_expr = tup
            if arg_name == 'text':
                if not isinstance(arg_expr, expr.ExprStr):
                    raise ValueError('Internal error: Expected an ExprStr')
                text = arg_expr.name
                del part.args_pre[i]

        # Prepare the horizontal offsets
        text_offsets = []
        for i in range(len(text)):
            c = max(0, min(95, ord(text[i]) - 32))
            for j in range(font_starts[c], font_starts[c + 1]):
                text_offsets.append(j)
        part.text_offsets = text_offsets # type: ignore

        # Load the image and attach it to the part
        part.image = load_image('font.png') # type: ignore
        part.half_wid = len(text_offsets) // 2 # type: ignore
        part.half_hgt = part.image.size[1] // 2 # type: ignore

    def render_pixel(self, outer_part:Optional[Part], blocks:Dict[str,Block], locals:Dict[str,ParamType], depth:int) -> List[float]:
        if depth >= 0:
            ind_print(2 * depth, f'type {self.name}')
            print_locals(2 * depth + 1, locals)
        x = cast(float, locals['x'])
        y = cast(float, locals['y'])
        z = cast(float, locals['z'])
        image = outer_part.image # type: ignore
        toff = outer_part.text_offsets # type: ignore
        hw = outer_part.half_wid # type: ignore
        hh = outer_part.half_hgt # type: ignore
        if x < -hw or y < -hh or x >= len(toff)-hw or y >= image.size[1]-hh:
            if depth >= 0:
                ind_print(2 * depth, f'miss')
            return [TOO_FAR, 0., 0., 0., 0.]
        else:
            gray = image.getpixel((toff[int(x) + hw], image.size[1] - 1 - (int(y) + hh)))
            if depth >= 0:
                ind_print(2 * depth, f'hit!')
            return [z, cast(float, locals['red']), cast(float, locals['green']), cast(float, locals['blue']), cast(float, locals['opacity']) * gray]




built_in_blocks = {
    'circle': BlockCircle(),
    'image': BlockImage(),
    'label': BlockLabel(),
    'square': BlockSquare(),
    'triangle': BlockTriangle(),
}

def find_block(name:str, blocks:Dict[str,Block], line:int) -> Block:
    if name in built_in_blocks:
        return built_in_blocks[name]
    if name in blocks:
        return blocks[name]
    if line > 0:
        raise ValueError(f'Error on line {line}: Type not found: {name}')
    else:
        raise ValueError(f'Type not found: {name}')

class FrameRenderer():
    def __init__(self, clip:Block, all_blocks:Dict[str,Block], frame:int, frame_count:int, out_height:int, in_width:int, in_height:int, one_row:bool=False) -> None:
        self.frame = frame
        self.frame_count = frame_count
        self.in_width = in_width
        self.in_height = in_height
        self.wid = (in_width * out_height) // in_height
        self.hgt = out_height
        self.half_wid = self.wid / 2
        self.half_hgt = self.hgt / 2
        self.scalar = in_height / out_height
        self.clip = clip
        self.all_blocks = all_blocks
        self.one_row = one_row
        if one_row:
            self.image = Image.new(mode="RGBA", size=(self.wid, 1))
        else:
            self.image = Image.new(mode="RGBA", size=(self.wid, self.hgt))
        self.args = {
            'z': 0.,
            't': frame / frame_count,
            'ft': 1. / frame_count,
        }

    def render_row(self, y:int) -> None:
        if self.one_row:
            yy = 0
        else:
            yy = y
        self.args['y'] = (y - self.half_hgt) * self.scalar
        for x in range(self.wid):
            self.args['x'] = (x - self.half_wid) * self.scalar
            _, r, g, b, opacity = self.clip.render_pixel(None, self.all_blocks, self.args, -1000) # type: ignore
            rgba = (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))), max(0, min(255, int(opacity * 256))))
            self.image.putpixel((x, self.hgt - 1 - yy), rgba)

    def debug_pixel(self, x:int, y:int) -> None:
        self.args['y'] = (self.hgt - 1 - y - self.half_hgt) * self.scalar
        self.args['x'] = (x - self.half_wid) * self.scalar
        z, r, g, b, opacity = self.clip.render_pixel(None, self.all_blocks, self.args, 0) # type: ignore
        print(f'dist={z} red={r} green={g} blue={b} opacity={opacity}')
