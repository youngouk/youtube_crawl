
import json
import os
import sys
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.storage.r2_storage import get_storage

def diagnose():
    storage = get_storage()
    state_file = root_dir / "data" / "state.json"
    if not state_file.exists():
        print("state.json not found")
        return
        
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
        
    # 1. Cost Estimation
    processed_videos = [v_id for v_id, s in state.items() if s.get('status') == 'completed']
    total_processed = len(processed_videos)
    
    # Average tokens (estimates based on typical YouTube transcripts)
    # Refine: 3000 input, 3000 output
    # Summary: 3000 input, 300 output
    # Context/Topics: 10 chunks/video * (2500 input, 150 output)
    
    avg_input_tokens = 3000 + 3000 + (10 * 2500) # 31,000 tokens
    avg_output_tokens = 3000 + 300 + (10 * 150) # 4,800 tokens
    
    total_input = total_processed * avg_input_tokens
    total_output = total_processed * avg_output_tokens
    
    # Gemini 1.5 Flash Pricing (approx)
    cost_input = (total_input / 1_000_000) * 0.075
    cost_output = (total_output / 1_000_000) * 0.30
    total_cost_usd = cost_input + cost_output
    
    print(f"--- Cost Estimate ---")
    print(f"Total Processed Videos: {total_processed}")
    print(f"Estimated Total Input Tokens: {total_input:,}")
    print(f"Estimated Total Output Tokens: {total_output:,}")
    print(f"Estimated Total Cost (USD): ${total_cost_usd:.2f}")
    print(f"Note: Using Gemini Free Tier (RPM 5) costs $0.00.")

    # 2. 삐뽀삐뽀 Channel Analysis
    channel_id = "UC6t0ees15Lp0gyrLrAyLeJQ"
    
    # Simulate 'both' fetching logic to see what videos we are dealing with
    # Or just read the latest saved list.json
    video_list_path = f"raw/videos/{channel_id}/list.json"
    channel_videos = storage.load_json(video_list_path) or []
    
    if not channel_videos:
        print(f"\n--- 삐뽀삐뽀 (UC6t0ees15Lp0gyrLrAyLeJQ) Analysis ---")
        print("No video list found in storage.")
        return

    print(f"\n--- 삐뽀삐뽀 (UC6t0ees15Lp0gyrLrAyLeJQ) Analysis ---")
    print(f"Total videos in current list.json: {len(channel_videos)}")
    
    completed = []
    failed = []
    not_processed = []
    
    for v in channel_videos:
        v_id = v['video_id']
        s = state.get(v_id, {})
        status = s.get('status', 'not_found')
        if status == 'completed':
            completed.append(v_id)
        elif status == 'failed':
            failed.append((v_id, s.get('error_message', 'No error msg')))
        else:
            not_processed.append((v_id, status))
            
    print(f"Status: Completed={len(completed)}, Failed={len(failed)}, Other={len(not_processed)}")
    if failed:
        print(f"First 5 failures:")
        for v_id, err in failed[:5]:
            print(f"  - {v_id}: {err}")
    if not_processed:
        print(f"First 5 others:")
        for v_id, stat in not_processed[:5]:
            print(f"  - {v_id}: {stat}")

if __name__ == "__main__":
    diagnose()
