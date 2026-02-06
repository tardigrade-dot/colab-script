import os
import re
import subprocess
import json
from pathlib import Path

MAX_DURATION_SECONDS = 8 * 3600 + 30 * 60
BLACK_CLIP_DURATION = 0.5
TARGET_FPS = 24


def run_ffmpeg(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def normalize_video(input_path, output_path):
    """统一所有段落的视频格式，使其使用固定FPS、统一编码和像素格式"""
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", f"fps={TARGET_FPS}",
        "-c:v", "libx264",          # 避免 videotoolbox 的 timestamp 问题
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path
    ])


def create_black_clip(path):
    """生成 0.5s 的黑屏 mp4，用于段落之间插入暂停"""
    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-f", "lavfi", "-i", f"color=black:s=1920x1080:d={BLACK_CLIP_DURATION}",
        "-vf", f"fps={TARGET_FPS}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        path
    ])


def get_video_duration(fp):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(fp)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def get_file_list_sorted(src_dir):
    regex = re.compile(r'.*-(\d+)_(\d+)\.mp4$')
    files = [f for f in os.listdir(src_dir) if regex.match(f)]
    files.sort(key=lambda f: tuple(map(int, regex.match(f).groups())))
    return [os.path.join(src_dir, f) for f in files]


def perform_concat(out_path, segment_paths):
    concat_list = out_path + "_concat.txt"
    with open(concat_list, "w") as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")

    run_ffmpeg([
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",              # 无损拼接
        out_path
    ])

# _norm.mp4 因为在生成视频时做了处理, 这里应该去掉这部分操作
def combine_mp4(mp4_dir):
    black_clip_path = os.path.join(mp4_dir, "black_0.5s_norm.mp4")
    if not os.path.exists(black_clip_path):
        print("生成统一格式黑屏视频...")
        create_black_clip(black_clip_path)

    # 预处理所有视频
    print(">>>> 正在统一视频格式")
    raw_files = get_file_list_sorted(mp4_dir)

    normalized_files = raw_files
    # normalized_files = []

    # for path in raw_files:
    #     out = path.replace(".mp4", "_norm.mp4")
    #     normalize_video(path, out)
    #     normalized_files.append(out)

    # 分段拼接
    print(">>>> 分段拼接")
    current, duration, part = [], 0.0, 1

    for path in normalized_files:
        d = get_video_duration(path)
        est = duration + d + (BLACK_CLIP_DURATION if current else 0)

        if est > MAX_DURATION_SECONDS and current:
            print(f"拼接 Part {part} ...")
            perform_concat(os.path.join(mp4_dir, f"result_part{part}.mp4"), current)
            part += 1
            current, duration = [path], d
        else:
            current.append(path)
            duration = est

    if current:
        print(f"拼接最后一段 Part {part} ...")
        perform_concat(os.path.join(mp4_dir, f"result_part{part}.mp4"), current)


if __name__ == "__main__":
    INPUT_MP4_DIR = "/Volumes/sw/tts_result/sulianjianshi"
    combine_mp4(INPUT_MP4_DIR)
    print("全部完成 🎉")
