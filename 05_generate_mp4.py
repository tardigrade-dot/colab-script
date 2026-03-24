# pip install moviepy pysrt tqdm

from moviepy import *
from tqdm import tqdm
import pysrt
import os
from pathlib import Path
import re
import argparse

# ---- 字体路径 ----
FONT_PATH = "/Users/larry/Documents/font/pingfang-sc-regular.ttf"

def wrap_text(text, max_chars_per_line=18):
    """
    按字数进行简单的自动换行，兼容中文。
    可根据实际字体和宽度微调 max_chars_per_line。
    """
    text = text.replace("\n", " ").strip()
    lines = []
    while len(text) > max_chars_per_line:
        lines.append(text[:max_chars_per_line])
        text = text[max_chars_per_line:]
    lines.append(text)
    return "\n".join(lines)

# ---- 主函数 ----
def process_single_wav(audio_path, image_path):

    wav_path = Path(audio_path)

    print(f'\nstart process file [{audio_path}] ...')
    output_path = str(wav_path.with_suffix('.mp4'))
    srt_path = str(wav_path.with_suffix('.srt'))

    if not wav_path.exists():
        print(f'⚠️ {str(wav_path)} not exists')
        return
    
    if not os.path.exists(srt_path):
        print(f'⚠️ {srt_path} not exists')
        return

    if os.path.exists(output_path):
        print(f'output file exists [{output_path}], not process')
        return
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    subs = pysrt.open(srt_path)

    # 背景
    video_w, video_h = 1280, 720
    bg_color = (245, 240, 230)  # 米色背景
    bg = ColorClip(size=(video_w, video_h), color=bg_color).with_duration(duration)

    # 图片区域（左边 10%-50%）
    img_clip = (
        ImageClip(image_path)
        .resized(height=int(video_h * 0.9))
        .with_position((int(video_w * 0.1), "center"))
        .with_duration(duration)
    )

    # 字幕区域（右边 50%-90%）
    text_area_w = int(video_w * 0.45)  # 宽度为总宽度的 40%
    text_x_pos = int(video_w * 0.5)   # 从 50% 开始
    max_chars_per_line = 14

    text_clips = []

    pre_end_time = 0
    for sub in subs:
        txt = sub.text.replace("\n", " ")
        start = sub.start.ordinal / 1000

        if pre_end_time !=0 and start < pre_end_time:
            start = pre_end_time
        end = sub.end.ordinal / 1000
        clip_duration = max(0.05, end - start)

        txt = wrap_text(sub.text, max_chars_per_line=max_chars_per_line)
        # 创建字幕片段
        try:
            text_clip = (
                TextClip(
                    text=txt + "\n",
                    font=FONT_PATH,
                    color="black",
                    font_size=36,
                    size=(text_area_w, 2000),  # 控制最大宽度，实现自动换行
                    # method="label",
                    method="caption", 
                )
                .with_start(start)
                .with_duration(clip_duration)
                .with_position((text_x_pos, "center"))
            )
        except Exception as e:
            print(f"[TextClip ERROR] {audio_path} | text: {txt} | error: {e}")
            raise Exception(e)
        text_clips.append(text_clip)
        pre_end_time = end

    final = (
        CompositeVideoClip([bg, img_clip] + text_clips)
        .with_audio(audio)
        .with_duration(duration)
    )

    final.write_videofile(
        output_path,
        fps=24,
        codec="h264_videotoolbox",
        audio_codec="aac",
        threads=8,
        ffmpeg_params=[
            "-vf", "fps=24",
            "-vsync", "cfr",
            "-video_track_timescale", "24000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-color_range", "mpeg",
            # "-strict", "-2"
        ]
    )


def get_file_list_sorted(wav_src_dir, wav_regex=None):

    if wav_regex is None:
        wav_regex = r'.*-(\d+)_(\d+).wav$'
    pattern = re.compile(wav_regex)
    files = [
        f for f in os.listdir(wav_src_dir)
        if pattern.match(f)
    ]

    def sort_key(filename):
        m = pattern.match(filename)
        if m:
            second, first = map(int, m.groups())
            return (second, first)
        return (float('inf'), float('inf'))

    sorted_files = sorted(files, key=sort_key)

    sorted_paths = [os.path.join(wav_src_dir, f) for f in sorted_files]
    return sorted_paths
    
import concurrent.futures
from pathlib import Path

def process_dir_batch(audio_dir, image_path, thread_num=1, wav_regex=None):
    """
    多线程处理指定目录下的所有 WAV 文件。

    Args:
        audio_dir (str | Path): 音频文件目录路径
        image_path (str | Path): 图像路径（所有音频共用同一张图，假设 process_single_wav 支持）
        wav_regex (str | None): 文件名匹配正则（可选）
        thread_num (int): 线程数，默认 1（即单线程）
    """
    # 获取所有待处理的 WAV 文件路径（保持顺序，但并行执行）
    file_paths = list(get_file_list_sorted(audio_dir, wav_regex))
    
    if not file_paths:
        print("No matching WAV files found.")
        return

    # 单线程走原逻辑（兼容 & 避免线程开销）
    if thread_num <= 1:
        for s_path in file_paths:
            process_single_wav(s_path, image_path)
        return

    # 多线程处理
    print(f"Processing {len(file_paths)} files with {thread_num} threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_single_wav, s_path, image_path)
            for s_path in file_paths
        ]

        # 可选：按完成顺序输出结果（或收集异常）
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 若 process_single_wav 有返回值可获取；否则仅为等待完成/捕获异常
            except Exception as e:
                # 推荐：记录出错的文件（可增强鲁棒性）
                # 注意：由于 submit 时未绑定路径，需额外传递信息
                # 更好的做法是用 callback 或包装函数
                print(f"Error processing a file: {e}")

def process_dir(audio_dir, image_path, wav_regex=None, thread_num=1):

    for s_path in get_file_list_sorted(audio_dir, wav_regex):
        
        process_single_wav(s_path, image_path)

def run_cli():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--dir_path", default="./hf_model", help="path to the directory containing wav files")
    parser.add_argument("--cover_path", type=str, default=0, help="cover image path")
    parser.add_argument("--batch", type=int, default=1, help="batch size for processing")

    args = parser.parse_args()
    process_dir_batch(args.dir_path, args.cover_path, args.batch)

def process(dir_or_file, cover_image):
    if os.path.isdir(dir_or_file):
        process_dir_batch(dir_or_file, cover_image)
    else:
        process_single_wav(dir_or_file, cover_image)


if __name__ == "__main__":

    # run_cli()
    process("/Volumes/sw/tts_result/renquandedixian/renquandedixian-2_0.wav", "/Volumes/sw/tts_result/renquandedixian.jpeg")
"""

/Volumes/sw/uv_envs/env_utils/bin/python /Users/larry/github.com/tardigrade-dot/colab-script/05_generate_mp4.py

"""


"""
!/Users/larry/miniconda3/envs/mix311/bin/python /Users/larry/github.com/tardigrade-dot/colab-script2/tools/moviepy_tool.py \
    --dir_path "/Volumes/sw/tts_result/tianchaoyaoyuan2" \
    --cover_path "/Volumes/sw/tts_result/tianchaoyaoyuan_cover_page2.png" \
    --batch 3

!/Users/larry/miniconda3/envs/mix311/bin/python /Users/larry/github.com/tardigrade-dot/colab-script2/tools/moviepy_tool.py \
    --dir_path "/Volumes/sw/tts_result/zhipeiyudikang" \
    --cover_path "/Volumes/sw/tts_result/zhipei-cover.jpg" \
    --batch 3
"""
