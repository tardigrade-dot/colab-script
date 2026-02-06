# sv311 安装SenseVoice环境: https://github.com/FunAudioLLM/SenseVoice

# pip install -r https://raw.githubusercontent.com/FunAudioLLM/SenseVoice/refs/heads/main/requirements.txt
# pip install wetext

import re
import os
from pathlib import Path
import string
from funasr import AutoModel
import difflib
from pydub import AudioSegment
from wetext import Normalizer

class CtcForcdAlign:
    def __init__(self, model_dir, device) -> None:

        self.MIN_CHARS_PER_LINE = 10 
        self.model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30_000},
            device=device,
            disable_update=True,
            trust_remote_code=True,
            # remote_code="./extro/model.py",
            remote_code="funasr.models.sense_voice.model",
        )

        self.normalizer = Normalizer(lang="zh", operator="tn", remove_erhua=True, traditional_to_simple=True)
        chinese_pattern = r"（.*?）"
        english_pattern = r"\([^)]*?\)"

        self.combined_pattern = f"{chinese_pattern}|{english_pattern}"
        self.PUNCTUATION_CHARS = set(string.punctuation) | set(',.?，。！？；：、“”‘’《》【】（）')
        self.PUNCT_REGEX = re.compile(r'[.,:;!?，。、；：？！“‘”’（）《》【】\s-]')
    
    def is_punctuation(self, token):
        return token in self.PUNCTUATION_CHARS
    
    def map_opcodes_to_raw(self, clean_opcodes, asr_timestamps, correct_tokens_raw):
        aligned_timestamps = []
        status = []
        
        j_raw_ptr = 0 
        
        for tag, i1, i2, j1, j2 in clean_opcodes:
            
            while j_raw_ptr < len(correct_tokens_raw) and self.PUNCT_REGEX.match(correct_tokens_raw[j_raw_ptr]):
                aligned_timestamps.append((None, None))
                j_raw_ptr += 1

            asr_chars_consumed = i2 - i1
            if tag == "equal":
                for k in range(j2 - j1): # 遍历匹配的汉字
                    aligned_timestamps.append(asr_timestamps[i1 + k])
                    status.append(True)
                    j_raw_ptr += 1 # 原始指针移动一个汉字
                    while j_raw_ptr < len(correct_tokens_raw) and self.PUNCT_REGEX.match(correct_tokens_raw[j_raw_ptr]):
                        aligned_timestamps.append((None, None))
                        j_raw_ptr += 1
            elif tag == "replace" or tag == "insert":
                for k in range(j2 - j1): # 遍历差异的汉字
                    if asr_chars_consumed > 0:
                        idx = i1 + k % asr_chars_consumed
                        aligned_timestamps.append(asr_timestamps[idx])
                    else: # 纯粹的插入 (tag='insert')
                        aligned_timestamps.append((None, None))
                    status.append(False)
                    j_raw_ptr += 1 # 原始指针移动一个汉字
                    while j_raw_ptr < len(correct_tokens_raw) and self.PUNCT_REGEX.match(correct_tokens_raw[j_raw_ptr]):
                        aligned_timestamps.append((None, None))
                        j_raw_ptr += 1
            elif tag == "delete":
                continue

        while j_raw_ptr < len(correct_tokens_raw) and self.PUNCT_REGEX.match(correct_tokens_raw[j_raw_ptr]):
            aligned_timestamps.append((None, None))
            j_raw_ptr += 1
        if len(aligned_timestamps) == len(correct_tokens_raw):
            return aligned_timestamps, status
        else:
            print(f"🚨 警告: 最终长度不匹配. Raw: {len(correct_tokens_raw)}, Aligned: {len(aligned_timestamps)}")
            return aligned_timestamps, status
        
    def map_asr_to_correct(self, asr_tokens, asr_timestamps, correct_tokens):
        
        punctuation_regex = r'[.,:;!?，。、；：？！“‘”’（）《》【】\s-]'
        clean_correct_tokens = re.sub(punctuation_regex, '', correct_tokens)
        matcher_clean = difflib.SequenceMatcher(None, asr_tokens, clean_correct_tokens)
        clean_opcodes = matcher_clean.get_opcodes()

        return self.map_opcodes_to_raw(clean_opcodes, asr_timestamps, correct_tokens)

    #old version
    def map_asr_to_correct_oldversion(self, asr_tokens, asr_timestamps, correct_tokens):
        """
        将ASR token时间戳映射到正确文本 token
        返回：
            aligned_timestamps: 对齐后的正确文本时间戳列表
            status: 每个 token 是否匹配
        """
        aligned_timestamps = []
        status = []

        # asr_tokens = "".join(asr_tokens)
        # 使用 difflib 对齐 ASR 与正确文本
        matcher = difflib.SequenceMatcher(self.is_punctuation, asr_tokens, correct_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # token 匹配 → 直接继承时间戳
                for k in range(j2 - j1):
                    aligned_timestamps.append(asr_timestamps[i1 + k])
                    status.append(True)
            elif tag == "replace":
                # token 替换 → 尝试平均分配时间戳
                # 可以选择标记为不匹配
                for k in range(j2 - j1):
                    if (i2 - i1) > 0:
                        idx = i1 + k % (i2 - i1)
                        aligned_timestamps.append(asr_timestamps[idx])
                    else:
                        # ASR 没有对应 token → 填充 None
                        aligned_timestamps.append((None, None))
                    status.append(False)
            elif tag == "insert":
                # 正确文本中新增 token → 无对应 ASR token
                for _ in range(j2 - j1):
                    aligned_timestamps.append((None, None))
                    status.append(False)
            elif tag == "delete":
                # ASR 中多余 token → 忽略
                continue

        return aligned_timestamps, status

    def is_valid_time(self, t):
        return isinstance(t, (int, float)) and t >= 0

    def get_valid_start_end(self, words):
        start_time, end_time = None, None

        for w in words:
            t = w.get('start')
            if self.is_valid_time(t):
                start_time = t
                break

        for w in reversed(words):
            t = w.get('end')
            if self.is_valid_time(t):
                end_time = t
                break

        return start_time, end_time

    def ms_to_srt_time_format(self, ms):
        # 确保输入是整数或可以转换为整数
        ms = int(ms)
        # 计算小时、分钟、秒和毫秒
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000

        # 使用 f-string 格式化输出，确保每个部分都有固定的位数
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def generate_srt_from_words_and_timestamps(self, words, timestamps):
        if not words or not timestamps or len(words) != len(timestamps):
            print(f"words length[{len(words)}] not equal to timestamps length[{len(timestamps)}]")
            raise Exception('process error')

        srt_content = ""
        subtitle_index = 1
        sentence_delimiters = ['。', '？', '！', '.', '?', '!', '…']
        
        current_subtitle_words = [] 
        end_time = 0
        
        # 内部函数：负责判断是否满足长度要求并输出字幕
        def _write_subtitle(words_to_write, pre_end_time):
            nonlocal srt_content, subtitle_index
            
            if not words_to_write:
                return None
                
            # start_time = words_to_write[0]['start']
            # end_time = words_to_write[-1]['end']
            try:

                start_time, end_time = self.get_valid_start_end(words_to_write)
                if pre_end_time and pre_end_time != 0 and start_time < pre_end_time:
                    start_time = pre_end_time
            except Exception as e:
                raise Exception(f"获取有效时间戳时出错, 音频质量存在问题: {e}")
            
            # 将单词列表组合成一个完整的句子
            sentence_text = "".join([w['word'] for w in words_to_write])
            
            srt_content += str(subtitle_index) + "\n"
            srt_content += f"{self.ms_to_srt_time_format(start_time)} --> {self.ms_to_srt_time_format(end_time)}\n"
            srt_content += sentence_text + "\n\n"
            subtitle_index += 1
            return end_time
        
        for i in range(len(words)):
            word = words[i]
            timestamp = timestamps[i]
            
            current_subtitle_words.append({'word': word, 'start': timestamp[0], 'end': timestamp[1]})
            
            is_sentence_end = word in sentence_delimiters
            if is_sentence_end:
                current_text = "".join([w['word'] for w in current_subtitle_words])
                current_length = len(current_text)
                if current_length >= self.MIN_CHARS_PER_LINE:
                    end_time = _write_subtitle(current_subtitle_words, end_time)
                    current_subtitle_words = [] # 清空累计列表
        if current_subtitle_words:
            all_none = all(
                item['start'] is None and item['end'] is None
                for item in current_subtitle_words
            )
            if not all_none:
                _write_subtitle(current_subtitle_words, end_time)
            
        return srt_content

    def generate_srt_file(self, wav_file, over_write=False):
        
        print(f'generate srt file for wav {wav_file}')
        wav_path = Path(wav_file)
        
        txt_path = wav_path.with_suffix('.txt')
        srt_path = wav_path.with_suffix('.srt')

        txt_file = str(txt_path)
        srt_file = str(srt_path)

        print(f'\nwav [{wav_file}], \ntxt [{txt_file}], \nsrt [{srt_file}]')

        if not wav_path.exists():
            raise Exception(f'wav file[{wav_file}] not exists ')
        if not txt_path.exists():
            raise Exception(f'txt file[{txt_file}] not exists ')
        if srt_path.exists() and not over_write:
            print(f'srt file[{srt_file}] exists, just return ')
            return
        
        with open(txt_file, 'r') as f:
            target_txt = "".join(f.readlines())

        res = self.model.generate(
                input=wav_file,
                cache={},
                language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                batch_size_s=30,
                merge_vad=True,
                merge_length_s=20,
                output_timestamp=True,
            )
        output = res[0]
        
        target_txt = re.sub(self.combined_pattern, "", target_txt)
        target_txt = target_txt.replace("\n", " ")
        target_txt = self.normalizer.normalize(target_txt)
        
        aligned_timestamps, status = self.map_asr_to_correct(output["words"], output["timestamp"], target_txt)

        num_tokens = len(status)
        num_mismatch = status.count(False)
        mismatch_ratio = num_mismatch / num_tokens

        if mismatch_ratio > 0.05:
            print(f"⚠️ ❌ 注意, 不匹配率过高.  不匹配 token({num_mismatch}/{num_tokens}) 占比: {mismatch_ratio:.2%}")
        else:
            print(f"⚠️ ✅ 不匹配 token({num_mismatch}/{num_tokens}) 占比: {mismatch_ratio:.2%}")

        srt_result = self.generate_srt_from_words_and_timestamps(target_txt, aligned_timestamps)
        
        with open(srt_file, 'w') as f:
            f.write(srt_result)
            
        print(f"字幕已生成, 保存在:{srt_file}")

    def get_wav_list_sorted(self, wav_src_dir, wav_regex=None):

        if wav_regex is None:
            wav_regex = r'.*-(\d+)_(\d+)\.wav$'
        pattern = re.compile(wav_regex)

        files = [
            f for f in os.listdir(wav_src_dir)
            if f.endswith('.wav') and pattern.match(f)
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

    def check_srt_exsis(self, wav_src_dir):
        srt_result = []
        for w_path in self.get_wav_list_sorted(wav_src_dir):

            srt_path = Path(w_path).with_suffix('.srt')

            if not srt_path.exists():
                srt_result.append(srt_path.stem)

        print(f'以下文件没有srt文件, 请检查语音文件 {srt_result}')

    def generate_srt_dir(self, wav_src_dir, over_write=False):
        for w_path in self.get_wav_list_sorted(wav_src_dir):

            try:
                self.generate_srt_file(w_path, over_write)
            except Exception as e:
                print(f"处理文件 {w_path} 时发生错误：{e}")

    def asr_with_target(self, wav_file):
        
        wav_path = Path(wav_file)
        txt_path = wav_path.with_suffix('.txt')
        
        output_dir = "/Volumes/sw/datasets_prepare/meiguodegushi_output/" + wav_path.stem
        if os.path.exists(output_dir):

            print(f'⚠️ output dir exists {output_dir}, nothing will process ...')
            return
        target = []
        with open(str(txt_path), 'r') as f:
            target = f.read().splitlines()

        res = self.model.generate(
            input=wav_file,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            batch_size_s=30,
            merge_vad=True,
            merge_length_s=20,
            output_timestamp=True,
        )
        output = res[0]
        sent_ts = output["timestamp_sentence"]

        target_txt = "".join(target)
        target_txt = self.normalizer.normalize(target_txt)

        target_txt = re.sub(self.combined_pattern, "", target_txt)
        target_txt = target_txt.replace("\n", "")

        target_tokens = self.model.kwargs['tokenizer'].text2tokens(target_txt)
        aligned_timestamps, status = self.map_asr_to_correct(output["words"], output["timestamp"], target_tokens)
        self.check_status(status)
        sen_clips = self.split_tokens_by_sentence(sent_ts, aligned_timestamps, target_tokens)
        self.export_audio_segments(wav_file, sen_clips, output_dir)

        print(f'✅ fininsh process wav {wav_file}')

    def export_audio_segments(self, audio_file, segments, output_dir):
        """
        audio_file: 原始音频路径
        segments: [{'text':..., 'start':..., 'end':...}, ...]
            start/end 单位：毫秒
        output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio = AudioSegment.from_file(audio_file)

        for i, seg in enumerate(segments, 1):
            start_ms = seg['start']
            end_ms = seg['end']
            clip = audio[start_ms:end_ms]
            # 保存 wav
            wav_path = os.path.join(output_dir, f"{i:03d}.wav")
            clip.export(wav_path, format="wav")

            # 保存对应文本
            txt_path = os.path.join(output_dir, f"{i:03d}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(seg['text'])

            # print(f"导出: {wav_path}, {txt_path}")

    def merge_short_sentences(self, sentences, min_duration=3000):
        """
        sentences: [{'text':..., 'start':..., 'end':...}, ...]
        min_duration: 最小时间阈值，单位与 start/end 一致（示例中为毫秒）
        """
        if not sentences:
            return []

        merged = [sentences[0]]

        for s in sentences[1:]:
            duration = s['end'] - s['start']
            if duration < min_duration:
                # 合并到上一条
                prev = merged[-1]
                prev['text'] += s['text']
                prev['end'] = s['end']
            else:
                merged.append(s)

        return merged

    def merge_sent_timestamps(self, sent_ts):
        """将每句中 token 的时间范围合并为 (start, end)"""
        if len(sent_ts) % 2 != 0:
            raise Exception('sent ts count not match with start and end timestamp')
        merged = []
        for i in range(0, len(sent_ts), 2):
            pair = sent_ts[i:i+2]  # 取两元素
            start = pair[0][0]
            end = pair[-1][1]
            merged.append((start, end))
        return merged

    def split_tokens_by_sentence(self, sent_ts, aligned_timestamps, target_tokens):
        assert len(aligned_timestamps) == len(target_tokens), "时间戳与token数量不一致"
        results = []
        idx = 0
        n = len(target_tokens)
        for sent_start, sent_end in sent_ts:
            sent_tokens = []
            # 遍历 token，直到 token 起点超过句子结束
            while idx < n:
                token_start, token_end = aligned_timestamps[idx]
                if token_start is not None and token_end is not None:
                    # token 完全在句子区间内
                    if token_start >= sent_start and token_end <= sent_end:
                        sent_tokens.append(target_tokens[idx])
                        idx += 1
                    # token 超出当前句子区间，说明当前句子结束
                    elif token_start >= sent_end or token_end <= sent_start:
                        break
                else:
                    sent_tokens.append(target_tokens[idx])
                    idx += 1

            if sent_tokens:
                sentence = "".join(sent_tokens)
                results.append({
                    "text": sentence,
                    "start": sent_start,
                    "end": sent_end
                })

        return results

    def check_status(self, status):
        num_tokens = len(status)
        num_mismatch = status.count(False)
        mismatch_ratio = num_mismatch / num_tokens

        if mismatch_ratio > 0.3:
            print(f"❌❌ 注意, 不匹配率过高.  不匹配 token({num_mismatch}/{num_tokens}) 占比: {mismatch_ratio:.2%}")
            raise Exception()
        if mismatch_ratio > 0.15:
            print(f"⚠️❌ 注意, 不匹配率过高.  不匹配 token({num_mismatch}/{num_tokens}) 占比: {mismatch_ratio:.2%}")
        else:
            print(f"⚠️ 不匹配 token({num_mismatch}/{num_tokens}) 占比: {mismatch_ratio:.2%}")

    def create_clip_with_asr(self, audio_dir):
        audio_files = sorted(os.listdir(audio_dir))
        for _a_f in audio_files:

            if _a_f.endswith('.wav'):
                
                _audio_file = os.path.join(audio_dir, _a_f)

                res_dir =Path(_audio_file).with_suffix("")
                res_dir.mkdir(parents=True, exist_ok=True)

                res = self.model.generate(
                    input=_audio_file,
                    cache={},
                    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=False,
                    batch_size_s=30,
                    merge_vad=True,
                    merge_length_s=20,
                    output_timestamp=True,
                )
                output = res[0]

                audio = AudioSegment.from_file(_audio_file)
                sent_timestamps = output['timestamp_sentence']
                words = output['words']
                words_ts = output['timestamp']

                for _i, sent_ts in enumerate(sent_timestamps):
                    _start_ts = sent_ts[0]
                    _end_ts = sent_ts[1]
                    clip = audio[_start_ts: _end_ts]

                    _clip_txt = []
                    for w, (w_s_ts, w_e_ts) in zip(words, words_ts):
                        if w_s_ts >= _start_ts and w_e_ts <= _end_ts:
                            _clip_txt.append(w)
                        if w_s_ts > _end_ts:
                            break
                    clip_txt = "".join(_clip_txt)

                    clip_path = res_dir / f"{_i:03d}.txt"
                    clip_path.write_text(clip_txt, encoding="utf-8")

                    wav_path = os.path.join(res_dir, f"{_i:03d}.wav")
                    clip.export(wav_path, format="wav")

                print(f'✅ ✅ ✅  文件{_a_f}处理完成')

env = "local" # colab local
if env == "colab":
    cfa = CtcForcdAlign(
        # model_dir = "/Volumes/sw/pretrained_models/SenseVoiceSmall",
        model_dir = "iic/SenseVoiceSmall",
        device = "cuda"
    )

    book_name = "tianchaoyaoyuan2"
    cfa.generate_srt_dir(f"/content/drive/MyDrive/{book_name}")
elif env == 'local':
    cfa = CtcForcdAlign(
        model_dir = "/Volumes/sw/pretrained_models/SenseVoiceSmall",
        device = "mps"
    )

    # cfa.generate_srt_file("/Volumes/sw/tts_result/fubaiyufanfu/fubaiyufanfu-3_1.wav", True)
else:
    raise Exception('env not support ')

book_name = "fubaiyufanfu"
cfa.generate_srt_dir(f"/Volumes/sw/tts_result/{book_name}", over_write=True)
cfa.check_srt_exsis(f"/Volumes/sw/tts_result/{book_name}")

# cfa.generate_srt_dir('/Volumes/sw/MyDrive/zhengzhi1/output', r"zhengzhi1-(\d+)_(\d+)\.wav", True)
# cfa.generate_srt_file("/Users/larry/github.com/tardigrade-dot/SenseVoice/data-forcedaligner/data.wav", True)

# wav_dir = "/Volumes/sw/datasets_prepare/zhongdong"
# for wav_file in os.listdir(wav_dir):
#     if wav_file.endswith('.wav'):
#         wav_path = os.path.join(wav_dir, wav_file)
#         cfa.asr_with_target(wav_path)

# cfa.asr_with_target("/Volumes/sw/datasets_prepare/meiguodegushi/真人朗读有声书《美国的故事》从1517宗教改革到美国建国400多年的风风雨雨 p05 05.美洲大陆的主人.mp3")
# cfa.generate_srt_file("/Volumes/sw/MyDrive/zhengzhi1/output/zhengzhi1-1_3.wav", True)

# cfa.create_clip_with_asr("/Volumes/sw/datasets_prepare/qiangpao")

# cfa.generate_srt_dir("/Users/larry/Documents/epub/quanliyuwuzhi", r"quanliyuwuzhi-(\d+)_(\d+)\.wav",)

# cfa.generate_srt_dir("/Users/larry/Documents/epub/fubaiyufanfu", r"fubaiyufanfu-(\d+)_(\d+)\.wav")

# cfa.generate_srt_file("/Users/larry/github.com/tardigrade-dot/colab-script2/data_src/zhipei-0_0.wav")

