
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.storage.r2_storage import R2Storage
from src.processors.chunker import Chunker

def analyze_transcripts():
    print("Connecting to R2 Storage...")
    try:
        storage = R2Storage()
    except Exception as e:
        print(f"Failed to connect to R2: {e}")
        return

    # List some transcript folders
    # Note: R2Storage doesn't have list_objects exposed directly in the interface I saw, 
    # but it initializes self.s3 (boto3 client).
    
    print("Listing transcripts...")
    try:
        response = storage.s3.list_objects_v2(Bucket=storage.bucket, Prefix='transcripts/', MaxKeys=20)
    except Exception as e:
        print(f"Failed to list objects: {e}")
        return

    if 'Contents' not in response:
        print("No transcripts found.")
        return

    files_to_analyze = []
    seen_videos = set()

    for obj in response['Contents']:
        key = obj['Key']
        # structure: transcripts/{video_id}/refined.json
        parts = key.split('/')
        if len(parts) >= 3 and parts[2] == 'refined.json':
            video_id = parts[1]
            if video_id not in seen_videos:
                files_to_analyze.append(key)
                seen_videos.add(video_id)
                if len(files_to_analyze) >= 3: # Analyze 3 samples
                    break
    
    if not files_to_analyze:
        print("No refined.json files found in the first 20 objects. Searching deeper...")
        # Try to find at least one
        paginator = storage.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=storage.bucket, Prefix='transcripts/')
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if 'refined.json' in key:
                    files_to_analyze.append(key)
                    if len(files_to_analyze) >= 3:
                        break
            if len(files_to_analyze) >= 3:
                break

    print(f"Analyzing {len(files_to_analyze)} transcript files...\n")

    chunk_sizes_to_test = [300, 150, 100, 50]

    for file_key in files_to_analyze:
        print(f"--- File: {file_key} ---")
        try:
            data = storage.load_json(file_key)
            if not data or 'text' not in data:
                print("Invalid data or no text field.")
                continue
            
            text = data['text']
            char_len = len(text)
            word_len = len(text.split())
            
            print(f"Total Characters: {char_len}")
            print(f"Total Words (spaces): {word_len}")
            if word_len > 0:
                print(f"Avg chars per word: {char_len / word_len:.2f}")

            print("\nSimulation:")
            for size in chunk_sizes_to_test:
                chunker = Chunker(chunk_size=size, chunk_overlap=int(size*0.15)) # 15% overlap
                chunks = chunker.split_text(text)
                print(f"  [Chunk Size {size}]: {len(chunks)} chunks generated.")
                if chunks:
                    # Print first chunk length sample
                    print(f"    - Sample Chunk 1 length: {len(chunks[0])} chars")
            print("")

        except Exception as e:
            print(f"Error processing {file_key}: {e}")

if __name__ == "__main__":
    analyze_transcripts()
