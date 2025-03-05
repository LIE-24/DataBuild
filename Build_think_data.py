import pandas as pd
import numpy as np
import re
import json
import time
import concurrent.futures
import os
from openai import OpenAI
from tqdm import tqdm

# pip install pandas numpy openai tqdm
# Define different token lengths
TOKEN_LENGTHS = [250,500,750,1000]
SAMPLES_PER_LENGTH = 1000
MAX_WORKERS =30
MAX_RETRIES = 5
SAVE_INTERVAL = 5  # Increased to reduce frequent saves
OPENAI_API_KEY = "sk-CS5hE81idQPPGgopcb5VpcpxRJOKyKaQwtMGtMG7GjZdbnKIwrjc5zPJnLT7vwG3"


def extract_thinking(content):
    """Extract thinking content between outermost <think> tags"""
    start_tag = '<think>'
    end_tag = '</think>'


    start = content.find(start_tag)
    if start == -1:
        print(f"Failed to extract thinking from: {content[:50]}...")
        return None


    start_content = start + len(start_tag)
    end = content.rfind(end_tag)
    if end == -1 or end < start_content:
        print(f"Failed to extract thinking from: {content[:50]}...")
        return None


    inner_content = content[start_content:end]
    print("成功提取到内容：", inner_content[:100])
    return inner_content.strip()


def compress_thinking(thinking_content, target_tokens):
    """Compress thinking content using DeepSeek API"""
    prompt = f"""Please compress the following content to not exceed {target_tokens} tokens while:
1. Preserving the original reasoning steps and logic
2. Maintaining the original style, tone, and expression patterns
3. Keeping the same approach to problem-solving
4. Retaining key calculations and important details
5. Ensuring the compressed version leads to the same conclusion

Content to compress:
{thinking_content}"""

    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(
                api_key="xxx",  # Use environment variable or replace with your API key
                base_url="xxx"
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a master of style mimicry and thought compression, specialized in preserving the exact reasoning patterns, voice, and mathematical approach of the original author."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            compressed = response.choices[0].message.content.strip()
            if not compressed:
                print(f"API returned empty content for: {thinking_content[:50]}...")
                return thinking_content  # Return original if empty
            print(f"Compressed to: {compressed[:50]}... (target: {target_tokens} tokens)")
            return compressed
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print("Max retries reached, returning original content")
                return thinking_content


def process_row(row, target_tokens, row_idx):
    try:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()

        modified = False
        think_replaced = False
        compression_instruction = f"The thinking between <think> tags should be compressed to not exceed {target_tokens} tokens"


        for i, msg in enumerate(messages):
            content = msg.get('content', '')

            if msg.get('role') == 'assistant' and '<think>' in content:
                start_tag = '<think>'
                end_tag = '</think>'
                start = content.find(start_tag)
                end = content.find(end_tag)

                if start != -1 and end != -1:
                    original = content[start + len(start_tag):end]
                    compressed = compress_thinking(original, target_tokens)
                    if compressed:
                        new_content = content.replace(
                            start_tag + original + end_tag,
                            start_tag + compressed + end_tag
                        )
                        messages[i]['content'] = new_content
                        modified = True
                        think_replaced = True
                        print("替换成功！")

        # 只有在 think 内容替换成功后，才添加用户消息的指令
        if think_replaced:
            for i, msg in enumerate(messages):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if not content.startswith(compression_instruction):
                        messages[i]['content'] = f"{compression_instruction}\n\n{content}"
                        modified = True

        if modified:
            row_copy = row.copy()
            row_copy['messages'] = messages
            return row_idx, row_copy
        return row_idx, None

    except Exception as e:
        print(f"Error processing row {row_idx}: {str(e)}")
        return row_idx, None


def process_parquet_file(input_file, output_base_file):
    """Process the parquet file with concurrency"""
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    start_idx = 0
    for target_tokens in TOKEN_LENGTHS:
        print(f"\nProcessing for target_tokens = {target_tokens}")
        end_idx = min(start_idx + SAMPLES_PER_LENGTH, len(df))

        if start_idx >= len(df):
            print(f"No more rows to process for {target_tokens} tokens")
            break


        output_file = f"{output_base_file.replace('.parquet', '')}_{target_tokens}.parquet"
        print(f"Will save to output file: {output_file}")


        subset_df = df.iloc[start_idx:end_idx].copy()
        modified_rows = []
        indices_to_update = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_row, row.to_dict(), target_tokens, i): i
                for i, (_, row) in enumerate(subset_df.iterrows())
            }

            processed_count = 0
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(subset_df),
                               desc=f"Processing {target_tokens} tokens"):
                row_idx, modified_row = future.result()
                if modified_row:

                    subset_df.at[subset_df.index[row_idx], 'messages'] = modified_row['messages']
                    processed_count += 1

                    if processed_count % SAVE_INTERVAL == 0:
                        subset_df.to_parquet(output_file)
                        print(f"Saved {processed_count} processed rows to {output_file}")


        if processed_count > 0:
            subset_df.to_parquet(output_file)
            print(f"Saved final {processed_count} processed rows to {output_file}")
        else:
            print(f"No rows were modified for token length {target_tokens}")
            subset_df.to_parquet(output_file)


        start_idx = end_idx

    print("Processing complete for all token lengths")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process parquet file to compress thinking content")
    parser.add_argument("--input",
                        default="train-00009-of-00010.parquet",
                        help="Input parquet file path (default: input.parquet)")
    parser.add_argument("--output",
                        default="train-00009-of-00010.parquet-build",
                        help="Output base file path without extension (default: output)")
    args = parser.parse_args()

    process_parquet_file(args.input, args.output)
