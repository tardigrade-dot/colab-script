from pathlib import Path
import re
import subprocess
import os
from tqdm import tqdm

# ================= 配置区 =================

FONT_FILE = "./data/pingfang-sc-regular.ttf"         # 放在脚本同目录下的字体文件名
FONT_NAME = "萍方-简"

MAX_CHARS_PER_LINE = 20        # 每行最多字符，超过自动换行

TARGET_FPS = 24
# 颜色与布局
BG_COLOR = "#F5F5DC"           # 米色背景
HIGHLIGHT_COLOR = "&H000000FF" # ASS格式红色 (BGR: &H00BBGGRR)
NORMAL_COLOR = "&H00888888"    # 灰色
CENTER_X = 1400                # 字幕水平中心位置（右侧）
CENTER_Y = 540                 # 字幕垂直中心位置
LINE_HEIGHT = 160              # 行间距（需大于字号以防重叠）

LINE_SPACING = 20    # 额外行间距（两块文本块之间的物理距离）
FONT_SIZE_HIGH = 62  # 对应 Style: High 的字号
FONT_SIZE_DEF = 48   # 对应 Style: Default 的字号

# 估算单行高度 (字号 * 1.2 左右通常是比较合适的行高)
SINGLE_LINE_H_HIGH = FONT_SIZE_HIGH * 1.2 
SINGLE_LINE_H_DEF = FONT_SIZE_DEF * 1.2
# ==========================================

def srt_time_to_ass(srt_time):
    h, m, s_ms = srt_time.split(':')
    s, ms = s_ms.split(',')
    return f"{int(h)}:{m}:{s}.{ms[:2]}"

def wrap_text(text, limit):
    """处理中文自动换行"""
    text = text.strip().replace('\n', '')
    if len(text) <= limit:
        return text
    # 按照限制字数切分并加入 ASS 换行符 \N
    return "\\N".join([text[i:i+limit] for i in range(0, len(text), limit)])

def get_text_height(text, single_line_h):
    """根据换行符计算文本块的总高度"""
    lines = text.count("\\N") + 1
    return lines * single_line_h

def generate_ass(srt_path, ass_path):
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920", "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColor, SecondaryColor, OutlineColor, BackColor, Bold, Italic, Underline, Strikeout, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{FONT_NAME},{FONT_SIZE_DEF},{NORMAL_COLOR},&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1",
        f"Style: High,{FONT_NAME},{FONT_SIZE_HIGH},{HIGHLIGHT_COLOR},&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\r\n', '\n')
    
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n$|$)')
    matches = pattern.findall(content)
    
    events = []
    move_t = 300 

    for i in range(len(matches)):
        start = srt_time_to_ass(matches[i][1])
        end = srt_time_to_ass(matches[i][2])
        
        # 预处理当前、上、下三卷文本
        curr_raw = wrap_text(matches[i][3], MAX_CHARS_PER_LINE)
        curr_h = get_text_height(curr_raw, SINGLE_LINE_H_HIGH)

        # 1. 当前句：从下方滑入中心 (红色)
        # 滑动距离：当前句高度的一半 + 间距 + 下一句高度的一半
        # 为了简化，我们假设滑动的偏移量为一个动态值 offset
        
        # 1. 当前句：始终在 CENTER_Y
        move_up = f"\\move({CENTER_X}, {CENTER_Y + 200}, {CENTER_X}, {CENTER_Y}, 0, {move_t})"
        events.append(f"Dialogue: 2,{start},{end},High,,0,0,0,,{{{move_up}}}{curr_raw}")

        # 2. 下一句：动态计算位置
        if i < len(matches) - 1:
            next_raw = wrap_text(matches[i+1][3], MAX_CHARS_PER_LINE)
            next_h = get_text_height(next_raw, SINGLE_LINE_H_DEF)
            # 计算下一句的 Y = 中心位置 + (当前句高/2 + 下一句高/2 + 间距)
            pos_y_next = CENTER_Y + (curr_h / 2 + next_h / 2 + LINE_SPACING)
            events.append(f"Dialogue: 1,{start},{end},Default,,0,0,0,,{{\\pos({CENTER_X}, {pos_y_next})}}{next_raw}")

        # 3. 上一句：动态计算位置并滑出
        if i > 0:
            prev_raw = wrap_text(matches[i-1][3], MAX_CHARS_PER_LINE)
            prev_h = get_text_height(prev_raw, SINGLE_LINE_H_DEF)
            # 计算上一句初始位置 Y = 中心位置 - (当前句高/2 + 上一句高/2 + 间距)
            pos_y_prev = CENTER_Y - (curr_h / 2 + prev_h / 2 + LINE_SPACING)
            # 滑出目标位置 Y = pos_y_prev - 200 (向上移出)
            move_out = f"\\move({CENTER_X}, {pos_y_prev}, {CENTER_X}, {pos_y_prev - 100}, 0, {move_t})"
            events.append(f"Dialogue: 1,{start},{end},Default,,0,0,0,,{{{move_out}}}{prev_raw}")

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header) + '\n' + '\n'.join(events))

def run_ffmpeg(ass_file, wav_file, output_mp4, cover_img):
    
    # 滤镜链说明：
    # [1:v] 缩放左图 -> [0:v] 叠加在米色背景上 -> 最后挂载字幕（指定当前文件夹寻找字体）
    filter_complex = (
        f"[1:v]scale=750:-1[img];"
        f"[0:v][img]overlay=180:(H-h)/2,subtitles={ass_file}:fontsdir='.'"
    )

    # v2 将后续的合并前的norm参数转移到生成的部分.
    cmd = [
        'ffmpeg', '-y',
        # 1. 强制背景源的帧率与 TARGET_FPS 一致
        '-f', 'lavfi', '-i', f'color=c={BG_COLOR}:s=1920x1080:r={TARGET_FPS}', 
        '-i', cover_img,
        '-i', wav_file,
        '-filter_complex', filter_complex,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        # 2. 必须包含这行，确保所有片段像素格式统一
        '-pix_fmt', 'yuv420p', 
        # 3. 强制设置输出帧率，防止 filter_complex 导致帧率漂移
        '-r', f'{TARGET_FPS}', 
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '44100',  # 建议固定音频采样率，防止合并时音频失效
        '-shortest',
        output_mp4
    ]

    # v1
    # cmd = [
    #     'ffmpeg', '-y',
    #     '-f', 'lavfi', '-i', f'color=c={BG_COLOR}:s=1920x1080', # 输入0: 米色背景
    #     '-i', cover_img,                                        # 输入1: 左侧图片
    #     '-i', wav_file,                                        # 输入2: 音频
    #     '-filter_complex', filter_complex,
    #     '-c:v', 'libx264',
    #     '-preset', 'fast',
    #     '-crf', '18',
    #     '-c:a', 'aac',
    #     '-b:a', '192k',
    #     '-shortest',
    #     output_mp4
    # ]
    
    print(f"{wav_file} 正在开始合成视频，请稍候...")
    subprocess.run(cmd)

def process_single(srt_file, cover_img):

    srt_path = Path(srt_file)
    ass_file = str(srt_path.with_suffix(".ass"))
    wav_file = str(srt_path.with_suffix(".wav"))
    output_mp4 = str(srt_path.with_suffix(".mp4"))

    if not os.path.exists(FONT_FILE):
        print(f"错误：找不到字体文件 {FONT_FILE}，请确保它在脚本同目录下。")
    else:
        if os.path.exists(output_mp4):
            print(f'{output_mp4} file exists continue to next...')
            return
        generate_ass(srt_file, ass_file)
        run_ffmpeg(ass_file, wav_file, output_mp4, cover_img)

def get_file_list_sorted(wav_src_dir, wav_regex=None):

    if wav_regex is None:
        wav_regex = r'.*-(\d+)_(\d+).srt$'
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

def generate_mp4(process_dir_or_file, cover_img):

    if os.path.isdir(process_dir_or_file):
        srt_regex = r'.*-(\d+)_(\d+).srt$'
        file_paths = list(get_file_list_sorted(process_dir_or_file, srt_regex))
        for srt_file in tqdm(file_paths):
            process_single(srt_file, cover_img)
    else:
        process_single(process_dir_or_file, cover_img)

if __name__ == "__main__":

    # generate_mp4("/Users/larry/github.com/tardigrade-dot/colab-script/data/renquandedixian-0_0.srt", 
    #                "/Users/larry/github.com/tardigrade-dot/colab-script/data/cover.jpeg")
    generate_mp4("/Volumes/sw/tts_result/sulianjianshi", 
                  "/Volumes/sw/tts_result/sulianjianshi-cover.jpg")
    