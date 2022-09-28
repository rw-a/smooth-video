import ffmpeg
import numpy
import subprocess
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
    """The M2M processing happens here"""
    if __name__ == '__main__':
        frame1_np = frame1[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)
        frame2_np = frame2[:, :, ::-1].astype(numpy.float32) * (1.0 / 255.0)

        frame1_tensor = torch.FloatTensor(numpy.ascontiguousarray(frame1_np.transpose(2, 0, 1)[None, :, :, :])).cuda()
        frame2_tensor = torch.FloatTensor(numpy.ascontiguousarray(frame2_np.transpose(2, 0, 1)[None, :, :, :])).cuda()

        interpolated_frame_tensor = \
            netNetwork(frame1_tensor, frame2_tensor, [torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()])[0]
        interpolated_frame_np = (interpolated_frame_tensor.detach().cpu().numpy()[0, :, :, :].
                                 transpose(1, 2, 0)[:, :, ::-1] * 255.0).clip(0.0, 255.0).round().astype(numpy.uint8)

        return interpolated_frame_np


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process1(in_filename):
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
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
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)

    previous_frame = read_frame(process1, width, height)
    while True:
        print("Loading frame...")
        current_frame = read_frame(process1, width, height)
        if current_frame is None:
            write_frame(process2, previous_frame)
            break
        interpolated_frame = interpolate_frame(previous_frame, current_frame)
        write_frame(process2, previous_frame)
        write_frame(process2, interpolated_frame)
        previous_frame = current_frame

    process1.wait()
    process2.stdin.close()
    process2.wait()


if __name__ == '__main__':
    run(INPUT_VIDEO, OUTPUT_VIDEO)
