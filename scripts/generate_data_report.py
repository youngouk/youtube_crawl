
import os
import json
import sys
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config.settings import settings
from src.storage.r2_storage import get_storage

def generate_report():
    storage = get_storage()
    
    # Load channel state
    state_file = root_dir / "data" / "channel_state.json"
    if not state_file.exists():
        print("Error: data/channel_state.json not found.")
        return
    
    with open(state_file, 'r', encoding='utf-8') as f:
        channel_state = json.load(f)
        
    print("=" * 100)
    print(f"{'채널명':<30} | {'처리됨':<6} | {'실패':<6} | {'자막없음':<8} | {'인기영상 TOP 3 (조회수)'}")
    print("-" * 100)
    
    total_processed = 0
    total_no_transcript = 0
    
    for channel_id, state in channel_state.items():
        if channel_id == "UC123": continue # Skip test channel
        
        name = state.get('name', 'Unknown')
        processed = state.get('processed_videos', 0)
        failed = state.get('failed_videos', 0)
        no_transcript = state.get('no_transcript_videos', 0)
        
        total_processed += processed
        total_no_transcript += no_transcript
        
        # Fetch video list for titles and views
        video_list_path = f"raw/videos/{channel_id}/list.json"
        videos = storage.load_json(video_list_path) or []
        
        top_titles = []
        if videos:
            # Sort by views if not already
            sorted_videos = sorted(videos, key=lambda x: x.get('view_count', 0), reverse=True)
            for v in sorted_videos[:3]:
                title = v.get('title', 'No Title')[:20] + "..."
                views = v.get('view_count', 0)
                top_titles.append(f"{title}({views:,})")
        
        top_titles_str = ", ".join(top_titles) if top_titles else "N/A"
        
        print(f"{name[:30]:<30} | {processed:<6} | {failed:<6} | {no_transcript:<8} | {top_titles_str}")

    print("-" * 100)
    print(f"합계: 총 {total_processed}개 영상 처리 완료 (자막 없음: {total_no_transcript}개)")
    
    # Simple size estimation (since we can't easily walk all files without listing objects)
    # Each processed video has: raw transcript, refined transcript, metadata, chunks (approx 100-200KB per video total)
    # Plus raw list.json per channel.
    estimated_size_mb = (total_processed * 150) / 1024 # Approx 150KB per video
    print(f"예상 데이터 용량 (R2): 약 {estimated_size_mb:.2f} MB")
    print("=" * 100)

if __name__ == "__main__":
    generate_report()
