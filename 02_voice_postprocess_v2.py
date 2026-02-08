# 音频 响度

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 有声书增强器（Python + FFmpeg）
功能：
  - 三段均衡（削减低频 + 提升中高频）
  - 静音段智能规整（TTS 优化）
  - 响度标准化至 -16 LUFS
  - 支持单文件 / 批量处理
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import re

# ==================== 配置 ====================
# 默认参数（可全局修改，或传参覆盖）
DEFAULT_PARAMS = {
    "highpass_freq": 70,       # 略微调低，保留一点男声磁性
    "lowpass_freq": 11000,     # 调高，给高音留点呼吸空间
    "eq_low_freq": 200,
    "eq_low_gain": -2.0,       # 压低浑浊区
    "eq_mid_freq": 3000,       # 避开人耳最敏感的 4kHz 齿音区
    "eq_mid_gain": 0.8,        # 适度提升清晰度
    "eq_high_freq": 6000,      # 提升更高频段
    "eq_high_gain": 0.5,
    "silence_threshold": "-80dB",
    "silence_leave": 0.75,
    "loudness_target": -16.0,
    "sample_rate": 24000,
}
# =============================================

# 日志配置（仅控制台）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()


def build_ffmpeg_filter(params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    
    filters = [
        # 1. 基础清理
        f"highpass=f={p['highpass_freq']}",
        f"lowpass=f={p['lowpass_freq']}",
        
        # 2. 三段均衡优化
        # 压低 200Hz 附近的鼻音/浑浊感
        f"equalizer=f={p['eq_low_freq']}:width_type=h:width=100:g={p['eq_low_gain']}",
        # 提升 3kHz 增加语音明亮度（避开刺耳区）
        f"equalizer=f={p['eq_mid_freq']}:width_type=h:width=500:g={p['eq_mid_gain']}",
        # 5kHz 以上使用高架滤镜，而不是均衡器，听感更顺滑
        f"treble=g={p['eq_high_gain']}:f={p['eq_high_freq']}",
        
        # 3. 动态压缩 (Compand) - 关键步骤：压制尖锐峰值
        # 这里的参数能让声音更浑厚、扎实
        "compand=attacks=0:points=-80/-80|-40/-35|-20/-20|0/-15",
        
        # 4. 静音规整
        f"silenceremove=start_periods=1:start_duration=0.1:start_threshold={p['silence_threshold']}:"
        f"stop_periods=-1:stop_duration={p['silence_leave']}:stop_threshold={p['silence_threshold']}",
        
        # 5. 响度标准化
        # 单次调用建议加上 measured 模式（这里简化处理）
        f"loudnorm=I={p['loudness_target']}:TP=-1.5:LRA=11"
    ]
    
    return ",".join(filters)

def process_single_file(
    input_path: str,
    output_path: str,
    params: dict = None,
    overwrite: bool = False
) -> bool:
    """
    处理单个音频文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        params: 自定义参数（覆盖 DEFAULT_PARAMS）
        overwrite: 是否覆盖已存在文件
    
    Returns:
        bool: 是否成功
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    if not input_path.exists():
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return False
    
    if output_path.exists() and not overwrite:
        logger.info(f"⏭️  跳过（已存在）: {output_path.name}")
        return True
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建命令
    filter_chain = build_ffmpeg_filter(params)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_path),
        "-af", filter_chain,
        "-ar", str(DEFAULT_PARAMS["sample_rate"]),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"❌ FFmpeg 失败 ({input_path.name}): {result.stderr.strip()}")
            return False
        logger.info(f"✅ 完成: {output_path.name}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 超时（>1小时）: {input_path.name}")
        return False
    except Exception as e:
        logger.error(f"❌ 异常: {input_path.name} → {e}")
        return False


def _process_single_wrapper(args):
    """供多进程调用的包装函数"""
    input_path, output_path, params, overwrite = args
    return process_single_file(input_path, output_path, params, overwrite)


def get_sorted_wav_files(input_dir: str):
    """按 zhipeiyudikang-X_Y.wav 排序"""
    files = list(Path(input_dir).glob("*.wav"))
    
    def sort_key(p):
        m = re.search(r"-(\d+)_(\d+)\.wav$", p.name, re.IGNORECASE)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)
    
    return sorted(files, key=sort_key)


def process_directory(
    input_dir: str,
    num_workers: int = 3,
    params: dict = None,
    overwrite: bool = False
):
    """
    批量处理目录内所有 .wav 文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        num_workers: 并行进程数（默认 CPU-1）
        params: 全局参数
        overwrite: 是否覆盖已存在文件
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 获取排序后的文件列表
    all_files = get_sorted_wav_files(input_dir)
    if not all_files:
        logger.warning(f"⚠️  未找到 .wav 文件: {input_dir}")
        return
    
    # 构建任务列表（跳过已存在文件）
    tasks = []
    # for f in all_files:
    #     # out_path = output_dir / f.name
    #     out = f.with_name(f.stem + "_enhanced" + f.suffix)
    #     if out.exists() and not overwrite:
    #         logger.info(f"⏭️  跳过（已存在）: {f.name}")
    #     else:
    #         tasks.append((f, out, params, overwrite))

    for f in all_files:
        if f.name.count("original") >0 :
            continue
        backup_path = f.with_name(f"{f.stem}_original{f.suffix}")
        out = f 
        if backup_path.exists() and not overwrite:
            logger.info(f"⏭️  跳过（备份已存在）: {f.name}")
            continue
        try:
            f.rename(backup_path)
            tasks.append((backup_path, out, params, overwrite))
        except Exception as e:
            logger.error(f"❌ 重命名失败 {f.name}: {e}")
    
    
    logger.info(f"📁 总文件数: {len(all_files)} | 待处理: {len(tasks)}")
    if not tasks:
        logger.info("✅ 全部已完成！")
        return
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # 多进程处理
    logger.info(f"🚀 启动 {num_workers} 进程...")
    success_count = 0
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_process_single_wrapper, tasks),
            total=len(tasks),
            desc="处理中"
        ))
        success_count = sum(results)
    
    # 最终报告
    logger.info(f"\n✅ 成功: {success_count} / {len(tasks)}")


# =============== 使用示例 ===============
if __name__ == "__main__":
    # === 示例 1：处理单个文件 ===
    # process_single_file(
    #     input_path="/Volumes/sw/tts_result/renquandedixian/renquandedixian-0_0.wav",
    #     output_path="/Volumes/sw/tts_result/renquandedixian/renquandedixian-0_0-1.wav"
    # )
    
    # === 示例 2：批量处理目录 ===
    process_directory(
        input_dir="/Volumes/sw/tts_result/wangquanyudikangquan",
        num_workers=4,  # 根据 CPU 调整
        overwrite=False  # 设为 True 可强制重处理
    )