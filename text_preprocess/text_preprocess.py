"""
Text preprocessing script: Convert numbers in text to standard Chinese expressions.
Uses DeepSeek's OpenAI-compatible API with batch processing (10 sentences per batch).
"""

import os
import time
from openai import OpenAI
#

# pandoc '/Volumes/sw/books/自由与权力 ([英]阿克顿 [未知]) (Z-Library).epub' -o ziyouyuquanli.md \
#   --toc --toc-depth=3 \
#   --strip-comments \
#   --markdown-headings=atx \
#   --wrap=none \
#   --lua-filter=clean.lua

# Configuration
API_KEY = "your_deepseek_api_key_here"  # Replace with your actual API key
INPUT_FILE = "/Users/larry/Documents/docs/zhuanzhiquanliyuzhongguoshehui.txt"
OUTPUT_FILE = "/Users/larry/Documents/docs/zzqlyzgsh.txt"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 5


def create_client(api_key):
    """Create DeepSeek OpenAI client."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )
    return client


def convert_numbers_to_chinese(client, sentences):
    """
    Convert a batch of sentences to replace numbers with standard Chinese expressions.
    
    Args:
        client: OpenAI client
        sentences: List of sentences to process
        
    Returns:
        List of processed sentences
    """
    prompt = "请将以下文本中的阿拉伯数字转换为标准的中文表达（如：20世纪→二十世纪，70年代→七十年代，8:30→八点半）。只需输出转换后的结果，每行一句，保持原有顺序，不要添加任何额外说明。\n\n"
    prompt += "\n".join(sentences)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的文本转换助手，负责将阿拉伯数字转换为标准的中文表达。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            return result.split("\n")
            
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  Failed to process batch after {MAX_RETRIES} retries, returning original sentences")
                return sentences


def process_file(input_file, output_file, batch_size=BATCH_SIZE):
    """
    Process input file line by line, convert numbers to Chinese expressions in batches.
    
    Args:
        input_file: Path to input text file
        output_file: Path to output processed text file
        batch_size: Number of sentences per batch request
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Read all lines from input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Total lines to process: {len(lines)}")
    
    # Initialize DeepSeek client
    client = create_client(API_KEY)
    
    processed_lines = []
    total_batches = (len(lines) + batch_size - 1) // batch_size
    
    for i in range(0, len(lines), batch_size):
        batch_num = (i // batch_size) + 1
        batch_lines = lines[i:i + batch_size]
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (lines {i+1}-{i+len(batch_lines)})...")
        processed_batch = convert_numbers_to_chinese(client, batch_lines)
        processed_lines.extend(processed_batch)
        
        # Rate limiting: wait between batches
        if batch_num < total_batches:
            time.sleep(1)
    
    # Write processed lines to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_lines))
    
    print(f"\nProcessing complete! Output saved to '{output_file}'")


if __name__ == "__main__":
    # Replace API_KEY variable or set environment variable before running
    # export DEEPSEEK_API_KEY="your_key_here"
    if os.environ.get("DEEPSEEK_API_KEY"):
        API_KEY = os.environ["DEEPSEEK_API_KEY"]
    
    process_file(INPUT_FILE, OUTPUT_FILE)
