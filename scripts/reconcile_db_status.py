
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.storage.r2_storage import get_storage

def reconcile():
    storage = get_storage()
    
    # 1. 전역 상태(state.json) 로드 - 실제 성공 여부의 원천
    state_file = root_dir / "data" / "state.json"
    if not state_file.exists():
        print("Error: data/state.json not found.")
        return
    with open(state_file, 'r', encoding='utf-8') as f:
        global_state = json.load(f)
    
    # 성공한 비디오 ID 세트
    completed_vids = {v_id for v_id, s in global_state.items() if s.get('status') == 'completed'}
    
    # 2. 채널 목록 및 비디오 매핑 로드
    channels_file = root_dir / "data" / "channels.json"
    if not channels_file.exists():
        print("Error: data/channels.json not found.")
        return
    with open(channels_file, 'r', encoding='utf-8') as f:
        channels = json.load(f)

    print("=" * 80)
    print(f"{'채널명':<30} | {'DB 성공 (정밀)':<12} | {'상태'}")
    print("-" * 80)

    total_db_success = 0
    
    for ch in channels:
        ch_id = ch['channel_id']
        ch_name = ch['name']
        
        # 해당 채널의 비디오 리스트 로드 (R2 또는 로컬)
        video_list_path = f"raw/videos/{ch_id}/list.json"
        videos = storage.load_json(video_list_path) or []
        
        # 현재 채널의 비디오 중 DB에서 'completed'인 것만 카운트
        db_success_count = 0
        for v in videos:
            if v['video_id'] in completed_vids:
                db_success_count += 1
        
        total_db_success += db_success_count
        status_label = "✅ 완료" if db_success_count >= 50 else "🔄 수집중/부족"
        
        # 삐뽀삐뽀 채널은 별도 강조
        if ch_id == "UC6t0ees15Lp0gyrLrAyLeJQ":
            ch_name = f"⭐ {ch_name}"
            
        print(f"{ch_name[:30]:<30} | {db_success_count:<12} | {status_label}")

    print("-" * 80)
    print(f"전체 DB 기준 성공 영상 합계: {total_db_success}개")
    print(f"(* 중복 제거 및 실제 인덱싱 완료 기준 수치입니다.)")
    print("=" * 80)

if __name__ == "__main__":
    reconcile()
