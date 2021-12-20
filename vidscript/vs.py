from typing import Mapping, Any, Dict, List, Optional, Callable
import threading
import queue
import os
import sys
import traceback
from PIL import Image
import signal
from . import renderer, parse_script, webserver, Script

RenderCallback = Callable[[int, int, int, int, int], None]
SaveFrameCallback = Callable[[str, str, int, Image.Image], None]

init_vals:Optional[Mapping[str,Any]] = None
jobs_done:List[str] = []
now_rendering_frame = -1
now_rendering_row = 0
render_errors = []
worker_thread_started = False
task_queue:"queue.Queue[RenderTask]" = queue.Queue()
keep_working = True

class RenderTask():
    def __init__(self, frame_renderer:Optional[renderer.FrameRenderer], y:int, filename:str) -> None:
        self.frame_renderer = frame_renderer
        self.y = y
        self.filename = filename

    def process(self) -> None:
        global now_rendering_frame
        global now_rendering_row
        global jobs_done
        if self.frame_renderer is None:
            return
        now_rendering_frame = self.frame_renderer.frame
        now_rendering_row = self.y
        self.frame_renderer.render_row(self.y)
        if self.y + 1 == self.frame_renderer.hgt:
            self.frame_renderer.image.save(self.filename)
            jobs_done.append(self.filename)
            now_rendering_frame = -1

def worker_thread() -> None:
    while keep_working:
        task:RenderTask = task_queue.get()
        try:
            task.process()
        except Exception as e:
            task_queue.queue.clear()
            render_errors.append(str(e))
            print(e)
        task_queue.task_done()

def do_index(query:Mapping[str, Any], session:webserver.Session) -> str:
    s = [
        """
<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
</head>
<body>
<table cellpadding="10px">
<tr>
    <td valign="top">
        <div style="background-color:#e0f0f0;padding:10px;width:500px;height:90vh;overflow-y:scroll;">
            <canvas id="framesCanvas" width="400" height="7000" style="border:1px solid #cccccc;"></canvas>
        </div>
    </td>
    <td valign="top">
        <button id="stop">Stop</button>
    </td>
</tr></table>

<script>
let frames = [];
let framesCanvas = document.getElementById('framesCanvas');
let out_wid = 355;
let out_hgt = 200;
let margin = 5;

class Frame {
    constructor(index) {
        this.index = index;
        this.image = new Image();
        this.drawn = false;
    }

    isImageLoaded(){
        return this.image !== undefined && this.image.complete && this.image.naturalHeight !== 0;
    }

    setImage(filename) {
        this.image.src = filename;
        this.drawn = false;
    }

    draw(ctx) {
        if (!this.drawn) {
            let w = out_wid;
            let h = out_hgt;
            let x = margin;
            let y = (out_hgt + margin) * this.index;
            if (this.isImageLoaded()) {
                ctx.drawImage(this.image, x, y);
                ctx.strokeStyle = "#60a060";
                ctx.beginPath();
                ctx.rect(x-1, y-1, w+2, h+2);
                ctx.stroke();
                this.drawn = true;
            } else {
                ctx.strokeStyle = "#804020";
                ctx.beginPath();
                ctx.rect(x-1, y-1, w+2, h+2);
                ctx.stroke();
            }
            //ctx.font = "30px Arial";
            ctx.fillStyle = "#002040";
            ctx.fillText(this.index, out_wid + 2 * margin + 5, y + out_hgt / 2);
        }
    }
}

function drawFrames(frame, row) {
    let ctx = framesCanvas.getContext("2d");

    // Draw the frames
    for(let i = 0; i < frames.length; i++) {
        let frame = frames[i];
        frame.draw(ctx);
    }

    // Draw the progress bar
    let pos = (out_hgt + margin) * frame + row;
    ctx.fillStyle = "#c0a0a0";
    ctx.fillRect(margin + out_wid + margin, pos, 2, framesCanvas.height - pos);
    ctx.fillStyle = "#408060";
    ctx.fillRect(margin + out_wid + margin, 0, 2, pos);
}

function httpPost(payload) {
    return new Promise(function (onSuccess, onFailure) {
        const request = new XMLHttpRequest();
        request.onreadystatechange = function () {
            if (request.readyState === 4) {
                if (request.status === 200) {
                    // console.log(`response: ${request.responseText}`);
                    onSuccess(JSON.parse(request.responseText));
                } else {
                    if (request.status === 0 && request.statusText.length === 0)
                        onFailure('Connection failed');
                    else
                        onFailure(
                            'Server returned status ' + request.status + ', ' + request.statusText
                        );
                }
            }
        };
        request.open('post', 'ajax.html', true);
        //request.setRequestHeader('Brownie', `sid=${getUserId()}`);
        request.setRequestHeader('Content-Type', 'application/json');
        request.send(JSON.stringify(payload));
    });
}

function onStop(event) {
    let payload = {
        action: 'stop',
    };
    httpPost(payload).then(
        function(response) {
            console.log('Stopped');
        },
        function(error) {
            alert(error);
        }
    );
}

function request_updates() {
    let payload = {
        action: 'get_updates',
    };
    httpPost(payload).then(
        function(response) {
            let done = response.done;
            for (let filename of done) {
                let last_slash = filename.lastIndexOf('/');
                let tail = filename.substring(last_slash + 2);
                let index = parseInt(tail);
                let im = document.getElementById(`f${index}`);
                if (index > 100)
                    throw new Error(`index out of range: ${index}`);
                while (frames.length <= index) {
                    frames.push(new Frame(index));
                }
                let frame = frames[index];
                frame.setImage(`${filename}?nonce=${Math.floor(Math.random() * 1000000)}`);
            }
            drawFrames(response.frame, response.row);
            if (response.err !== undefined) {
                alert(response.err);
            }
            if (response.init !== undefined) {
                // document.getElementById('video_obj').value = response.init.video_obj;
                // document.getElementById('frame_count').value = response.init.frame_count;
                // document.getElementById('beg').value = response.init.beg;
                // document.getElementById('end').value = response.init.end;
                // document.getElementById('in_width').value = response.init.in_width;
                // document.getElementById('in_height').value = response.init.in_height;
                // document.getElementById('out_height').value = response.init.out_height;
                // document.getElementById('filename').value = response.init.script_name;
            }
        },
        function(error) {
            alert(error);
        }
    );
}

function onExit() {
    let payload = {
        action: 'shutdown',
    };
    httpPost(payload).then(
        function(response) {
            console.log('shut down');
        },
        function(error) {
            alert(error);
        }
    );
}

document.getElementById("stop").addEventListener("click", onStop);
window.onbeforeunload = onExit;

let timer = setInterval(function() { request_updates(); }, 500);

</script>
</body></html>
""",
    ]
    return ''.join(s)

def do_gui() -> None:
    # signal.signal(signal.SIGINT, signal_handler) # Register signal_handler to respond to Ctrl-C
    webserver.SimpleWebServer.render({
        '': do_index,
        'ajax.html': do_ajax,
        'index.html': do_index,
    })

def launch_rendering(
    all_blocks:Dict[str,renderer.Block],
    render_params:Dict[str,Any],
) -> Dict[str, Any]:
    global worker_thread_started
    global init_vals

    # Find the clip and frame ranges
    clip = renderer.find_block(render_params['video_obj'], all_blocks)
    if clip is None:
        return {
            'error': 'Bad arguments: No object named "{video_obj}"',
        }
    beg, end = render_params['beg'], render_params['end']
    if end < 0 or end > render_params['frame_count']:
        end = render_params['frame_count']
    if beg < 0 or beg >= end:
        beg = 0
    init_vals = render_params

    # Make the destination folder if needed
    script_base = os.path.splitext(os.path.basename(render_params['script_name']))[0]
    out_folder = os.path.join(script_base, render_params['video_obj'])
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Make sure the worker thread has been started
    if not worker_thread_started:
        print('(Press Ctrl-C twice to quit.)')
        worker_thread_started = True
        threading.Thread(target=worker_thread).start()

    # Make the rendering jobs
    for frame in range(beg, end):
        frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame, render_params['frame_count'], render_params['out_height'], render_params['in_width'], render_params['in_height'])
        dest_filename = f'{out_folder}/f{frame:07}.png'
        for y in range(frame_renderer.hgt):
            task_queue.put(RenderTask(frame_renderer, y, dest_filename))
    return { 'msg': 'started' }

def do_ajax(query:Mapping[str, Any], session:webserver.Session) -> Dict[str, Any]:
    global jobs_done
    global init_vals
    action = query['action']
    if action == 'get_updates':
        jobs_done, tmp = [], jobs_done
        packet = {
            'done': tmp,
            'frame': now_rendering_frame,
            'row': now_rendering_row,
        }
        if len(render_errors) > 0:
            packet['err'] = render_errors[0]
            del render_errors[0]
        if init_vals is not None:
            packet['init'] = init_vals
            init_vals = None
        return packet
    elif action == 'stop':
        task_queue.queue.clear()
        return { 'msg': 'ack' }
    elif action == 'shutdown':
        keep_working = False
        task_queue.queue.clear()
        task_queue.put(RenderTask(None, 0, '')) # No-op to flush out any threads waiting on the queue
        # task_queue.join()

        # Exit
        print('\nGoodbye.')
        sys.exit(0)
        # return { 'msg': 'ack' }
    else:
        return { 'error': f'unrecognized action: {action}' }

# def signal_handler(sig:int, frame) -> None: # type: ignore
#     # Join with the worker thread
#     keep_working = False
#     task_queue.queue.clear()
#     task_queue.put(RenderTask(None, 0, ''))
#     # task_queue.join()
#
#     # Exit
#     print('\nGoodbye.')
#     sys.exit(0)

def render_gui(
    script:Script,
    script_name:str,
    video_obj:str,
    frame_count:int=10,
    out_height:int=200,
    in_width:int=1600,
    in_height:int=900,
    beg:int=0,
    end:int=-1,
) -> None:
    retval = launch_rendering(
        script.to_blocks(),
        {
            'script_name': script_name,
            'video_obj': video_obj,
            'frame_count': frame_count,
            'out_height': out_height,
            'in_width': in_width,
            'in_height': in_height,
            'beg': beg,
            'end': end,
        }
    )
    if 'error' in retval:
        if 'stack' in retval:
            print(retval['stack'])
        raise ValueError(retval['error'])
    do_gui()

def defaultRenderCallback(beg:int, end:int, frame:int, out_height:int, y:int) -> None:
    if y % 10 == 0: # Don't waste time updating too often
        tot = (end - beg) * out_height
        pos = (frame - beg) * out_height + y
        print(f'  {100 * pos / tot:.2f}% (Frame {frame-beg}/{end-beg})', end='\r')

def defaultSaveFrameCallback(script_name:str, video_obj:str, frame:int, image:Image) -> None:
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    folder_name = os.path.join(script_base, video_obj)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    image.save(os.path.join(folder_name, f'f{frame:07}.png'))
    print(f'                            ', end='\r')

def render(
    script:Script,
    script_name:str,
    video_obj:str,
    frame_count:int=10,
    out_height:int=200,
    in_width:int=1600,
    in_height:int=900,
    beg:int=0,
    end:int=-1,
    renderCallback:RenderCallback=defaultRenderCallback,
    saveFrameCallback:SaveFrameCallback=defaultSaveFrameCallback,
) -> None:
    all_blocks = script.to_blocks()
    clip = renderer.find_block(video_obj, all_blocks)
    if clip is None:
        raise ValueError('Bad arguments: No object named "{video_obj}"')
    if end < 0 or end > frame_count:
        end = frame_count
    if beg < 0 or beg >= end:
        beg = 0
    for frame in range(beg, end):
        frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame, frame_count, out_height, in_width, in_height)
        for y in range(frame_renderer.hgt):
            renderCallback(beg, end, frame, out_height, y)
            frame_renderer.render_row(y)
        saveFrameCallback(script_name, video_obj, frame, frame_renderer.image)

    # Print a helpful message
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    folder_name = os.path.join(script_base, video_obj)
    print('Example command to compile the frames into an animated gif:')
    print(f'convert -delay 10 -loop 0 {folder_name}/f*.png {script_base}.gif')

# Render a single pixel (as specified in output coordinates, where 0,0 is the top-left pixel) for debugging purposes
def render_debug(
    script:Script,
    script_name:str,
    video_obj:str,
    frame_count:int=10,
    out_height:int=200,
    frame:int=0,
    output_x:int=100,
    output_y:int=100,
    in_width:int=1600,
    in_height:int=900
) -> None:
    all_blocks = script.to_blocks()
    clip = renderer.find_block(video_obj, all_blocks)
    if clip is None:
        raise ValueError('Bad arguments: No object named "{video_obj}"')
    frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame, frame_count, out_height, in_width, in_height)
    frame_renderer.debug_pixel(output_x, frame_renderer.hgt - 1 - output_y)

def parse(script:str, start_line:int=-1) -> Script:
    if start_line < 0:
        start_line = int(sys._getframe(1).f_lineno) # Get the line number of the caller
    lines = script.split('\n')
    return parse_script(lines, start_line)
