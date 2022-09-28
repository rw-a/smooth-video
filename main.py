import ffmpeg
import numpy as np
import subprocess
import PIL.Image
import torch
import model.m2m as m2m

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"

"""Load M2M Model"""
if not torch.cuda.is_available():
    raise Exception("CUDA GPU not detected. CUDA is required.")

torch.set_grad_enabled(False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

netNetwork.load_state_dict(torch.load('./model.pkl'))


def interpolate_frame(frame1, frame2):
    """The M2M Model sits here"""
    import shutil
    import os

    os.mkdir('input_frames')
    os.mkdir('output_frames')

    if __name__ == '__main__':
        npyOne = np.array(PIL.Image.open(f'input_frames/{str(1).zfill(8)}.png'))[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        npyTwo = np.array(PIL.Image.open(f'input_frames/{str(2).zfill(8)}.png'))[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)

        tenOne = torch.FloatTensor(np.ascontiguousarray(npyOne.transpose(2, 0, 1)[None, :, :, :])).cuda()
        tenTwo = torch.FloatTensor(np.ascontiguousarray(npyTwo.transpose(2, 0, 1)[None, :, :, :])).cuda()

        tenEstimate = netNetwork(tenOne, tenTwo, [torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()])[0]
        npyEstimate = (tenEstimate.detach().cpu().np()[0, :, :, :].transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).round().astype(np.uint8)

        img = PIL.Image.fromarray(npyEstimate[:, :, ::-1])  # reverses colour channels to get RGB

        shutil.copy2(f'input_frames/{str(1).zfill(8)}.png', f'output_frames/{str(1).zfill(8)}.png')
        img.save(f'output_frames/{str(2).zfill(8)}.png', format='png')


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process1(in_filename):
    print("Starting ffmpeg process1")
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    print("Starting ffmpeg process2")
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def write_frame(process2, frame):
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def run(in_filename, out_filename):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)

    previous_frame = read_frame(process1, width, height)
    while True:
        print("Loading frame...")
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            break
        out_frame = interpolate_frame(previous_frame, in_frame)
        write_frame(process2, previous_frame)
        write_frame(process2, out_frame)
        previous_frame = in_frame
    write_frame(process2, in_frame)

    process1.wait()
    process2.stdin.close()
    process2.wait()


if __name__ == '__main__':
    run(INPUT_VIDEO, OUTPUT_VIDEO)
