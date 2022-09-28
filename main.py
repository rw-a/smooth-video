import subprocess
import argparse
import ffmpeg
import numpy
import torch
import model.m2m as m2m

##########################################################
"""Options/Args"""

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, required=True, help="Input video name (e.g. input.mp4)")
parser.add_argument('-o', '--output', type=str, required=True, help="Output video name (e.g. output.mp4)")
parser.add_argument('-f', '--factor', type=int, default=2, help="Interpolation factor. 2 means double frame rate")

args = parser.parse_args()

if args.factor < 2:
    raise ValueError('Factor must be an integer more than or equal to 2.')

##########################################################
"""Load M2M Model"""

if not torch.cuda.is_available():
    raise Exception("CUDA GPU not detected. CUDA is required.")

torch.set_grad_enabled(False)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

netNetwork = m2m.M2M_PWC().cuda().eval()

netNetwork.load_state_dict(torch.load('./model.pkl'))

##########################################################


def interpolate_frame(frame1, frame2):
    """The M2M processing happens here"""
    frame1_np = frame1[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
    frame2_np = frame2[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)

    frame1_tensor = torch.FloatTensor(numpy.ascontiguousarray(frame1_np.transpose(2, 0, 1)[None, :, :, :])).cuda()
    frame2_tensor = torch.FloatTensor(numpy.ascontiguousarray(frame2_np.transpose(2, 0, 1)[None, :, :, :])).cuda()

    if args.factor == 2:
        interpolated_frame_tensor = \
            netNetwork(frame1_tensor, frame2_tensor, [torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()])[0]
        interpolated_frame_np = (interpolated_frame_tensor.detach().cpu().numpy()[0, :, :, :].
                                 transpose(1, 2, 0)[:, :, ::-1] * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)
        return interpolated_frame_np
    elif args.factor > 2:
        frames = []
        for i in range(args.factor - 1):
            interpolated_frame_tensor = \
                netNetwork(frame1_tensor, frame2_tensor, [torch.FloatTensor([(i + 1) / args.factor]).view(1, 1, 1, 1).
                           cuda()])[0]
            interpolated_frame_np = (interpolated_frame_tensor.detach().cpu().numpy()[0, :, :, :].
                                     transpose(1, 2, 0)[:, :, ::-1] * 255.0).clip(0.0, 255.0).round().astype(
                numpy.uint8)
            frames.append(interpolated_frame_np)
        return frames


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process_input(in_filename):
    process_args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(process_args, stdout=subprocess.PIPE)


def start_ffmpeg_process_output(out_filename, width, height):
    process_args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(process_args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            numpy
            .frombuffer(in_bytes, numpy.uint8)
            .reshape([height, width, 3])
        )
    return frame


def write_frame(process2, frame):
    process2.stdin.write(
        frame
        .astype(numpy.uint8)
        .tobytes()
    )


def run(in_filename, out_filename):
    width, height = get_video_size(in_filename)
    process_input = start_ffmpeg_process_input(in_filename)
    process_output = start_ffmpeg_process_output(out_filename, width, height)

    previous_frame = read_frame(process_input, width, height)
    while True:
        write_frame(process_output, previous_frame)
        print("Loading frame...")
        current_frame = read_frame(process_input, width, height)
        if current_frame is None:
            break
        if args.factor == 2:
            interpolated_frame = interpolate_frame(previous_frame, current_frame)
            write_frame(process_output, interpolated_frame)
        elif args.factor > 2:
            for interpolated_frame in interpolate_frame(previous_frame, current_frame):
                write_frame(process_output, interpolated_frame)
        previous_frame = current_frame

    process_input.wait()
    process_output.stdin.close()
    process_output.wait()


if __name__ == '__main__':
    run(args.input, args.output)
