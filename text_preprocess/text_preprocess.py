"""
Text preprocessing script: Convert numbers in text to standard Chinese expressions.
Uses DeepSeek's OpenAI-compatible API. Only processes sentences containing numbers.
Splits text by punctuation to minimize sentence size for model processing.
"""

import os
import re
import time
from openai import OpenAI

# Configuration
API_KEY = "your_deepseek_api_key_here"  # Replace with your actual API key
INPUT_FILE = "/Users/larry/Documents/docs/ziyouyuquanli.md"
OUTPUT_FILE = "/Users/larry/Documents/docs/ziyouyuquanli2.md"
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 5

# Punctuation marks for splitting sentences
SENTENCE_SPLIT_PATTERN = r'[，。！？；、,.!?;：:\n\r]'


def create_client(api_key):
    """Create DeepSeek OpenAI client."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )
    return client


def has_number(text):
    """Check if text contains any Arabic numerals."""
    return bool(re.search(r'\d', text))


def split_text_by_punctuation(text):
    """
    Split text into sentences by punctuation marks.
    Keeps punctuation attached to sentences.
    
    Args:
        text: Input text (can be a paragraph or multi-line text)
    
    Returns:
        List of sentence fragments with punctuation
    """
    # Split by punctuation but keep the punctuation
    sentences = re.split(f'({SENTENCE_SPLIT_PATTERN})', text)
    
    # Reattach punctuation to sentences
    result = []
    current_sentence = ""
    
    for part in sentences:
        if re.match(SENTENCE_SPLIT_PATTERN, part):
            # This is a punctuation mark, attach to current sentence
            current_sentence += part
            if current_sentence.strip():
                result.append(current_sentence.strip())
            current_sentence = ""
        elif part.strip():
            # This is text content
            current_sentence += part
    
    # Don't forget the last sentence if it doesn't end with punctuation
    if current_sentence.strip():
        result.append(current_sentence.strip())
    
    return result


def contains_chinese(text):
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def convert_numbers_to_chinese(client, sentences):
    """
    Convert a batch of sentences to replace numbers with standard Chinese expressions.

    Args:
        client: OpenAI client
        sentences: List of sentences to process

    Returns:
        List of processed sentences
    """
    prompt = "请将以下文本中的阿拉伯数字转换为标准的中文表达（如:20世纪→二十世纪,70年代→七十年代,8:30→八点半）。只需输出转换后的结果,每行一句,保持原有顺序,不要添加任何额外说明。\n\n"
    prompt += "\n".join(sentences)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的文本转换助手,负责将阿拉伯数字转换为标准的中文表达。"},
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


def process_text(client, text, batch_size=BATCH_SIZE):
    """
    Process a single text/paragraph: split by punctuation, convert numbers in sentences.
    Only sends sentences containing numbers to the model.

    Args:
        client: OpenAI client
        text: Input text string
        batch_size: Number of sentences per batch request

    Returns:
        Processed text with numbers converted to Chinese expressions
    """
    # Split text into sentence fragments
    fragments = split_text_by_punctuation(text)
    
    if not fragments:
        return text
    
    print(f"  Split into {len(fragments)} fragments")
    
    # Identify which fragments contain numbers (only these need model processing)
    numbered_fragments = []
    fragment_indices = []  # Track original indices
    
    for i, fragment in enumerate(fragments):
        if has_number(fragment):
            numbered_fragments.append(fragment)
            fragment_indices.append(i)
    
    print(f"  Found {len(numbered_fragments)} fragments with numbers")
    
    if not numbered_fragments:
        # No numbers in this text, return as-is
        return text
    
    # Process numbered fragments in batches
    processed_numbered = []
    total_batches = (len(numbered_fragments) + batch_size - 1) // batch_size
    
    for i in range(0, len(numbered_fragments), batch_size):
        batch_num = (i // batch_size) + 1
        batch_fragments = numbered_fragments[i:i + batch_size]
        
        print(f"  Processing batch {batch_num}/{total_batches}...")
        processed_batch = convert_numbers_to_chinese(client, batch_fragments)
        processed_numbered.extend(processed_batch)
        
        # Rate limiting
        if batch_num < total_batches:
            time.sleep(1)
    
    # Reconstruct the text with processed fragments
    # Replace original numbered fragments with processed ones
    result_fragments = fragments.copy()
    for idx, processed_frag in enumerate(processed_numbered):
        original_idx = fragment_indices[idx]
        result_fragments[original_idx] = processed_frag
    
    # Join fragments back together
    return "".join(result_fragments)


def process_file(input_file, output_file, batch_size=BATCH_SIZE):
    """
    Process input file: split by punctuation, convert numbers in batches.
    Handles paragraphs and maintains paragraph structure.

    Args:
        input_file: Path to input text file
        output_file: Path to output processed text file
        batch_size: Number of sentences per batch request
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read entire file content
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split content into paragraphs (by double newlines or single newlines)
    paragraphs = re.split(r'\n\s*\n', content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    print(f"Total paragraphs to process: {len(paragraphs)}")

    # Initialize DeepSeek client
    client = create_client(API_KEY)

    processed_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        print(f"\nProcessing paragraph {i+1}/{len(paragraphs)}...")
        processed_para = process_text(client, paragraph, batch_size)
        processed_paragraphs.append(processed_para)
        
        # Rate limiting between paragraphs
        if i < len(paragraphs) - 1:
            time.sleep(1)

    # Write processed paragraphs to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(processed_paragraphs))

    print(f"\nProcessing complete! Output saved to '{output_file}'")


if __name__ == "__main__":
    # Replace API_KEY variable or set environment variable before running
    # export DEEPSEEK_API_KEY="your_key_here"
    if os.environ.get("DEEPSEEK_API_KEY"):
        API_KEY = os.environ["DEEPSEEK_API_KEY"]

    process_file(INPUT_FILE, OUTPUT_FILE)
