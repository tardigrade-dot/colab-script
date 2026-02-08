import os
import re
from pathlib import Path
from typing import Union, List

from wetext import Normalizer

# 初始化 Normalizer 实例（在全局或函数内部一次性完成）
#### 效果有限. 有些转换结果不好, 比如几十年, 转换为了几10年
try:
    NORMALIZER = Normalizer(
        lang="zh",
        operator="itn",
        remove_erhua=True,
        traditional_to_simple=False,
        enable_0_to_9=False
    )
    print("Normalizer 初始化成功。")
except Exception as e:
    print(f"错误：初始化 wetext.Normalizer 失败。请检查 wetext 是否正确安装。")
    print(f"详细错误: {e}")
    NORMALIZER = None


def _get_itn_processed_lines(lines: List[str], normalizer: Normalizer) -> List[str]:
    """
    内部辅助函数：对 SRT 文件中的文本行进行 ITN 处理。
    
    Args:
        lines: 原始 SRT 文件的所有行。
        normalizer: 已初始化的 Normalizer 实例。
        
    Returns:
        处理后的所有行。
    """
    processed_lines = []
    
    for line in lines:
        # SRT 文件结构判断逻辑：
        # 非空行
        # 且不是纯数字（SRT 序号）
        # 且不包含 '-->'（SRT 时间轴）
        is_subtitle_text = line.strip() and not re.match(r'^\d+$', line.strip()) and '-->' not in line

        if is_subtitle_text:
            # 使用 ITN 进行反规范化
            normalized_text = normalizer.normalize(line.strip())

            # 特殊处理：将“〇”替换回“0”
            normalized_text = normalized_text.replace("〇", "0") 
            processed_lines.append(normalized_text + '\n')
        else:
            processed_lines.append(line)
            
    return processed_lines


def process_single_srt_itn(
    file_path: Union[str, Path], 
    num1: Union[int, float] = float('inf'), 
    num2: Union[int, float] = float('inf')
):
    """
    处理单个 .srt 文件，进行 ITN 反规范化。
    
    核心逻辑：直接读取原文件，处理后覆盖写入原文件。

    Args:
        file_path: 待处理的 SRT 文件的完整路径。
        num1: 文件的第一个排序序号，用于打印日志。
        num2: 文件的第二个排序序号，用于打印日志。
    """
    if NORMALIZER is None:
        print("❌ 错误：Normalizer 未成功初始化，无法进行处理。")
        return

    input_file = Path(file_path)
    if not input_file.is_file() or input_file.suffix.lower() != '.srt':
        print(f"❌ 错误：文件不存在或不是一个 .srt 文件：{file_path}")
        return

    # --- 日志打印 ---
    if num1 == float('inf'):
        print(f"\n正在处理未排序文件：{input_file.name}")
    else:
        print(f"\n正在处理文件 (序号: {num1}-{num2})：{input_file.name}")
    
    
    # 输出文件就是原文件
    output_file = input_file 

    try:
        # 1. 读取原始 SRT 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 2. 处理字幕内容
        processed_lines = _get_itn_processed_lines(lines, NORMALIZER)

        # 3. 将处理后的内容覆盖写入到原文件路径
        # 注意: 覆盖写入时，如果文件很大，需要考虑内存和磁盘I/O的效率，但对于普通字幕文件影响不大。
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(processed_lines)
        
        print(f"-> 成功！ITN 结果已覆盖原文件：{output_file.name}")

    except FileNotFoundError:
        print(f"❌ 错误：文件 '{input_file.name}' 未找到，无法进行处理。")
    except Exception as e:
        print(f"❌ 处理文件 {input_file.name} 时发生错误：{e}")


def process_directory_srt_itn(directory_path: str):
    """
    遍历指定目录下的所有 .srt 文件，使用 ITN 反规范化，
    并按照 'str-num1_num2.srt' 格式进行排序处理。
    
    Args:
        directory_path: 包含 SRT 文件的目录路径。
    """
    if NORMALIZER is None:
        print("❌ 错误：Normalizer 未成功初始化，无法进行处理。")
        return

    target_dir = Path(directory_path)
    if not target_dir.is_dir():
        print(f"❌ 错误：目录不存在或不是一个有效目录：{directory_path}")
        return

    print(f"\n--- 开始批量处理目录：{target_dir} ---")

    # 正则表达式匹配 'str-num1_num2.srt' 格式中的 num1 和 num2
    filename_pattern = re.compile(r'(\d+)_(\d+)\.srt$', re.IGNORECASE)

    srt_files_with_keys = []
    
    # 查找所有 .srt 文件并提取排序键
    for input_file in target_dir.glob("*.srt"):
        match = filename_pattern.search(input_file.name)
        if match:
            # 提取 num1 和 num2，转换为整数作为排序键
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            srt_files_with_keys.append((num1, num2, input_file))
        else:
            # 对于不符合命名格式的文件，可以默认放在最后处理
            srt_files_with_keys.append((float('inf'), float('inf'), input_file))

    # 按照 num1, num2 进行排序
    srt_files_with_keys.sort(key=lambda x: (x[0], x[1]))
    
    if not srt_files_with_keys:
        print("未找到任何 .srt 字幕文件。")
        return

    # 遍历排序后的文件，并调用单文件处理函数
    for num1, num2, input_file in srt_files_with_keys:
        # 核心：将文件处理逻辑委托给 process_single_srt_itn
        process_single_srt_itn(input_file, num1=num1, num2=num2)


if __name__ == "__main__":
    # 请根据您的实际环境修改此路径
    target_directory = "/Volumes/sw/tts_result/sulianjianshi" 
    print(f"当前脚本处理目录：{target_directory}")
    
    # 批量处理目录
    process_directory_srt_itn(target_directory)
    
    # 单文件测试示例 (请确保文件存在)
    # file_to_process = Path(target_directory) / "zhipeiyudikang-0_0.srt"
    # if os.path.exists(file_to_process):
    #     print("\n--- 独立测试单个文件 ---")
    #     process_single_srt_itn(file_to_process)
    # else:
    #     print(f"\n跳过单文件测试，因为文件 '{file_to_process.name}' 不存在。")