from typing import Mapping, Any, Dict, List, Optional, Callable, Tuple, cast
import threading
import queue
import os
import sys
import traceback
from PIL import Image
import pathlib
import requests
from . import renderer, parse_script, webserver, Script

RenderCallback = Callable[[int, int, int, int, int], None]
SaveFrameCallback = Callable[[str, str, int, Image.Image], None]

init_vals:Dict[str,Any] = {}
jobs_done:List[str] = []
completed_row_count = 0
completed_rows:List[bool] = []
render_errors:List[str] = []
script_to_render:Script = Script()
worker_thread_started = False
task_queue:"queue.Queue[RenderTask]" = queue.Queue()
keep_working = True
frame_renderers:Dict[int, renderer.FrameRenderer] = {}

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

# def signal_handler(sig:int, frame) -> None: # type: ignore
#     # Join with the worker thread
#     keep_working = False
#     task_queue.queue.clear()
#     task_queue.put(RenderTask(None, 0, '', 0))
#     # task_queue.join()
#
#     # Exit
#     print('\nGoodbye.')
#     sys.exit(0)








class RenderTask():
    def __init__(self, frame_renderer:Optional[renderer.FrameRenderer], y:int, filename:str, beg:int) -> None:
        self.frame_renderer = frame_renderer
        self.y = y
        self.filename = filename
        self.beg = beg

    def process(self) -> None:
        global jobs_done
        global completed_rows
        global completed_row_count
        if self.frame_renderer is None:
            return
        self.frame_renderer.render_row(self.y)

        # Advance the completed rows
        completed_index = (self.frame_renderer.frame - self.beg) * self.frame_renderer.hgt + self.y
        completed_rows[completed_index] = True
        prev_completed_row_count = completed_row_count
        while completed_row_count < len(completed_rows) and completed_rows[completed_row_count]:
            completed_row_count += 1
            if completed_row_count % self.frame_renderer.hgt == 0:
                self.frame_renderer.image.save(self.filename)
                jobs_done.append(self.filename)

def worker_thread() -> None:
    while keep_working:
        task:RenderTask = task_queue.get()
        try:
            task.process()
        except Exception as e:
            traceback.print_exc()
            task_queue.queue.clear()
            render_errors.append(str(e))
        task_queue.task_done()

def defaultRenderCallback(beg:int, end:int, frame:int, out_height:int, y:int) -> None:
    if y % 10 == 0: # Don't waste time updating too often
        tot = (end - beg) * out_height
        pos = (frame - beg) * out_height + y
        print(f'  {100 * pos / tot:.2f}% (Frame {frame-beg}/{end-beg})', end='\r')

def defaultFilename(script_name:str, video_obj:str, frame:int) -> str:
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    folder_name = os.path.join(script_base, video_obj)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    return os.path.join(folder_name, f'f{frame:07}.png')

def defaultSaveFrameCallback(script_name:str, video_obj:str, frame:int, image:Image) -> None:
    image.save(defaultFilename(script_name, video_obj, frame))
    print(f'                            ', end='\r')

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
    passive:bool=True, # unused
    renderCallback:RenderCallback=defaultRenderCallback, # unused
    saveFrameCallback:SaveFrameCallback=defaultSaveFrameCallback, # unused
    frame:int=0, # unused
    output_x:int=100, # unused
    output_y:int=100, # unused
) -> None:
    global worker_thread_started
    global init_vals
    global script_to_render
    global completed_rows
    global frame_renderers

    # signal.signal(signal.SIGINT, signal_handler) # Register signal_handler to respond to Ctrl-C

    # Find the clip and frame ranges
    all_blocks = script.to_blocks()
    clip = renderer.find_block(video_obj, all_blocks, 0)
    if end < 0 or end > frame_count:
        end = frame_count
    if beg < 0 or beg >= end:
        beg = 0

    # Values to pass to the front end
    init_vals = {
        'script_name': script_name,
        'video_obj': video_obj,
        'frame_count': frame_count,
        'out_height': out_height,
        'in_width': in_width,
        'in_height': in_height,
        'beg': beg,
        'end': end,
    }

    # Values to pass to the render thread
    completed_rows = [False] * (end - beg) * out_height
    script_to_render = script

    # Make the destination folder if needed
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    out_folder = os.path.join(script_base, video_obj)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Make sure the worker thread has been started
    if worker_thread_started:
        task_queue.queue.clear()
        frame_renderers = {}
    else:
        print('(Press Ctrl-C twice to quit.)')
        worker_thread_started = True
        threading.Thread(target=worker_thread).start()

    # Make the rendering jobs
    for frame_index in range(beg, end):
        frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame_index, frame_count, out_height, in_width, in_height)
        frame_renderers[frame_index] = frame_renderer
        dest_filename = f'{out_folder}/f{frame_index:07}.png'
        for y in range(frame_renderer.hgt):
            task_queue.put(RenderTask(frame_renderer, y, dest_filename, beg))

    # Launch the GUI
    webserver.SimpleWebServer.render({
        '': do_index,
        'ajax.html': do_ajax,
        'index.html': do_index,
    })

def do_ajax(query:Mapping[str, Any], session:webserver.Session) -> Dict[str, Any]:
    global jobs_done
    global init_vals
    global completed_row_count
    global keep_working
    action = query['action']
    if action == 'get_updates':
        jobs_done, tmp = [], jobs_done
        if 'beg' in init_vals and 'out_height' in init_vals:
            hgt = init_vals['out_height']
            beg = init_vals['beg']
            completed_frame = hgt * beg + completed_row_count // hgt
            completed_row = completed_row_count % hgt
        else:
            raise ValueError('init_vals should be set before the gui is started')
        packet = {
            'done': tmp,
            'frame': completed_frame,
            'row': completed_row,
        }
        if len(render_errors) > 0:
            packet['err'] = render_errors[0]
            del render_errors[0]
        return packet
    elif action == 'stop':
        task_queue.queue.clear()
        return { 'msg': 'ack' }
    elif action == 'shutdown':
        keep_working = False
        task_queue.queue.clear()
        task_queue.put(RenderTask(None, 0, '', 0)) # No-op to flush out any threads waiting on the queue
        # task_queue.join()

        # Exit
        print('\nGoodbye.')
        sys.exit(0)
        # return { 'msg': 'ack' }
    elif action == 'worker': # A worker wants a task
        if 'results' in query:
            # Receive results
            frame = query['frame']
            y = query['y']
            results = query['results']
            frame_renderer = frame_renderers[frame]
            img = frame_renderer.image
            assert img.size[1] > y and img.size[0] == len(results), f'Results do not match expectations. Expected {img.shape}. Got {len(results)},{len(results[0]) if len(results) > 0 else 0}'
            for x in range(img.size[0]):
                img.putpixel((x, y), tuple(results[x]))

        # Grab the next job
        if not keep_working:
            return {} # All done
        task:RenderTask = task_queue.get()
        task_queue.task_done() # Ideally, we should wait until the task is really done to do this, but meh.
        response:Dict[str,Any] = {}

        # If there are no more jobs to do
        if task.frame_renderer is None:
            keep_working = False
            try:
                task_queue.queue.clear()
            except Exception as e:
                traceback.print_exc()
            return response # No more jobs to do

        # If this is a brand new worker
        if not 'results' in query:
            # Send the whole script to render
            response['script'] = str(script_to_render)

        # Give the worker another row to render
        response['video_obj'] = init_vals['video_obj']
        response['frame'] = task.frame_renderer.frame
        response['frame_count'] = task.frame_renderer.frame_count
        response['out_height'] = task.frame_renderer.hgt
        response['in_width'] = task.frame_renderer.in_width
        response['in_height'] = task.frame_renderer.in_height
        response['y'] = task.y
        return response
    else:
        return { 'error': f'unrecognized action: {action}' }

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
    passive:bool=True, # if true, it will render only if they file does not already exist
    renderCallback:RenderCallback=defaultRenderCallback,
    saveFrameCallback:SaveFrameCallback=defaultSaveFrameCallback,
    frame:int=0, # unused
    output_x:int=100, # unused
    output_y:int=100, # unused
) -> None:
    all_blocks = script.to_blocks()
    clip = renderer.find_block(video_obj, all_blocks, 0)
    if end < 0 or end > frame_count:
        end = frame_count
    if beg < 0 or beg >= end:
        beg = 0
    for frame_index in range(beg, end):
        if passive:
            filename = defaultFilename(script_name, video_obj, frame_index)
            if os.path.exists(filename):
                continue
            else:
                pathlib.Path(filename).touch() # so other passive processes won't try to render this frame while we work on it
        frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame_index, frame_count, out_height, in_width, in_height)
        for y in range(frame_renderer.hgt):
            renderCallback(beg, end, frame_index, out_height, y)
            frame_renderer.render_row(y)
        saveFrameCallback(script_name, video_obj, frame_index, frame_renderer.image)

    # Print a helpful message
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    folder_name = os.path.join(script_base, video_obj)
    print('Example command to compile the frames into an animated gif:')
    print(f'convert -delay 10 -loop 0 {folder_name}/f*.png {script_base}.gif')
    print('')
    print('Example command to compile the frames into a mov:')
    print(f"ffmpeg -framerate 10 -pattern_type glob -i '{folder_name}/*.png' \-c:v prores -pix_fmt yuva444p10le {script_base}.mov")

# Render a single pixel (as specified in output coordinates, where 0,0 is the top-left pixel) for debugging purposes
def render_debug(
    script:Script,
    script_name:str,
    video_obj:str,
    frame_count:int=10,
    out_height:int=200,
    in_width:int=1600,
    in_height:int=900,
    beg:int=0, # unused
    end:int=-1, # unused
    passive:bool=True, # unused
    renderCallback:RenderCallback=defaultRenderCallback, # unused
    saveFrameCallback:SaveFrameCallback=defaultSaveFrameCallback, # unused
    frame:int=0,
    output_x:int=100,
    output_y:int=100,
) -> None:
    print(f'Debugging {script_name}:')
    all_blocks = script.to_blocks()
    clip = renderer.find_block(video_obj, all_blocks, 0)
    frame_renderer = renderer.FrameRenderer(clip, all_blocks, frame, frame_count, out_height, in_width, in_height)
    frame_renderer.debug_pixel(output_x, frame_renderer.hgt - 1 - output_y)

def parse(script:str, start_line:int=-1) -> Script:
    if start_line < 0:
        start_line = int(sys._getframe(1).f_lineno) # Get the line number of the caller
    lines = script.split('\n')
    return parse_script(lines, start_line)

def worker(server_url:str='http://localhost:8986/ajax.html') -> None:
    # Tell the server we are ready to work
    all_blocks:Dict[str,renderer.Block] = {}
    payload:Dict[str,Any] = {
        'action': 'worker',
    }
    req = requests.post(server_url, json=payload)

    # Parse the script from the server
    response = req.json()
    print('---script---')
    print(response['script'])
    script = parse(response['script'])
    all_blocks = script.to_blocks()
    clip = renderer.find_block(response['video_obj'], all_blocks, 0)

    # Start rendering
    while True:
        if not 'y' in response:
            break # The server has no row for us to render, so we're done

        # Render the specified row of pixels
        frame_renderer = renderer.FrameRenderer(clip, all_blocks, response['frame'], response['frame_count'], response['out_height'], response['in_width'], response['in_height'])
        frame_renderer.render_row(response['y'])

        # Convert to an array of ints
        results:List[Tuple[int,int,int,int]] = []
        for x in range(frame_renderer.image.size[0]):
            results.append(frame_renderer.image.getpixel((x, 0)))

        # Send back to the server (who is acting as master)
        payload = {
            'action': 'worker',
            'frame': response['frame'], # Tell the server which frame we just rendered
            'y': response['y'], # Tell the server which row we just rendered
            'results': results, # Send the row of pixels back to the server
        }
        req = requests.post(server_url, json=payload)
        response = req.json()
