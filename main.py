"""
Medical RAG 파이프라인 메인 실행 모듈입니다.

소아과 관련 YouTube 채널들을 순회하며 비디오를 수집하고,
트랜스크립트를 정제한 후 Pinecone에 인덱싱합니다.
"""
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Any
from config.settings import settings
from src.collectors.youtube_collector import YouTubeCollector
from src.collectors.transcript_collector import TranscriptCollector
from src.processors.gemini_processor import GeminiProcessor
from src.processors.chunker import Chunker
from src.storage.r2_storage import get_storage
from src.storage.state_manager import StateManager
from src.storage.channel_state_manager import ChannelStateManager
from src.vector_db.pinecone_manager import PineconeManager
from src.utils.logger import logger


def create_chunk_metadata(
    chunk_text: str,
    context: str,
    video_id: str,
    video_title: str,
    channel_id: str,
    channel_name: str,
    chunk_index: int,
    published_at: str,
    is_verified_professional: bool,
    specialty: str,
    credentials: str,
    timestamp_start: str = '',
    timestamp_end: str = '',
    topics: list[str] | None = None
) -> dict[str, Any]:
    """
    Pinecone 저장용 청크 메타데이터를 생성합니다.

    Args:
        chunk_text: 청크 텍스트
        context: Contextual Retrieval로 생성된 컨텍스트
        video_id: YouTube 비디오 ID
        video_title: 비디오 제목
        channel_id: YouTube 채널 ID
        channel_name: 채널 이름
        chunk_index: 청크 인덱스
        published_at: 비디오 게시일 (ISO 8601)
        is_verified_professional: 의료 전문가 인증 여부
        specialty: 전문 분야 (예: 소아과)
        credentials: 전문가 자격 정보 (예: 소아청소년과 전문의)
        timestamp_start: 청크 시작 시간 (예: "02:30")
        timestamp_end: 청크 종료 시간 (예: "04:15")
        topics: 의료 관련 토픽 키워드 리스트

    Returns:
        메타데이터 딕셔너리
    """
    return {
        'text': chunk_text,
        'context': context,
        'video_id': video_id,
        'video_title': video_title,
        'channel_id': channel_id,
        'channel_name': channel_name,
        'chunk_index': chunk_index,
        'source_type': 'youtube',
        'video_url': f'https://www.youtube.com/watch?v={video_id}',
        'published_at': published_at,
        'is_verified_professional': is_verified_professional,
        'specialty': specialty,
        'credentials': credentials,
        'timestamp_start': timestamp_start,
        'timestamp_end': timestamp_end,
        'topics': topics or [],
        'processed_at': datetime.now(timezone.utc).isoformat()
    }


def format_transcript_to_text(transcript_list: list[dict[str, Any]]) -> str:
    """Converts transcript list to a single string."""
    return " ".join([t['text'] for t in transcript_list])


def format_timestamp(seconds: float) -> str:
    """
    초 단위 시간을 MM:SS 형식으로 변환합니다.

    Args:
        seconds: 초 단위 시간 (예: 150.5)

    Returns:
        MM:SS 형식 문자열 (예: "02:30")
    """
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def normalize_specialty(specialty_value: Any) -> str:
    """
    specialty 필드를 문자열로 정규화합니다.

    channels.json에서 배열 또는 문자열로 저장될 수 있으므로
    일관된 문자열 형태로 변환합니다.

    Args:
        specialty_value: 배열 또는 문자열 형태의 specialty 값

    Returns:
        정규화된 문자열 (배열인 경우 첫 번째 값)
    """
    if isinstance(specialty_value, list):
        return specialty_value[0] if specialty_value else ''
    return str(specialty_value) if specialty_value else ''

def main(
    retry_failed: bool = False,
    skip_completed_channels: bool = True,
    specific_channel: str | None = None,
    max_results: int = 50,
    sort_override: str | None = None
) -> None:
    """
    파이프라인 메인 실행 루프입니다.
    """
    logger.info("=" * 60)
    logger.info("Starting Medical RAG Pipeline - Multi-Channel Processing")
    logger.info("=" * 60)
    
    if retry_failed:
        logger.info("Retry mode enabled: will retry failed videos")
    if specific_channel:
        logger.info(f"Processing specific channel only: {specific_channel}")
    
    # 정렬 및 개수 설정 로깅
    current_sort = sort_override or settings.VIDEO_SORT_BY
    logger.info(f"Sort mode: {current_sort}, Max results: {max_results}")

    # Initialize components
    storage = get_storage()
    state_manager = StateManager()
    channel_state_manager = ChannelStateManager()
    
    # Check needed keys
    if not settings.YOUTUBE_API_KEY:
        logger.error("YOUTUBE_API_KEY missing. Exiting.")
        return
        
    yt_collector = YouTubeCollector()
    transcript_collector = TranscriptCollector()
    
    try:
        gemini = GeminiProcessor()
    except Exception as e:
        logger.error(f"Failed to init Gemini: {e}")
        return

    # Pinecone is optional for initial collection/processing test
    pinecone_manager = None
    if settings.PINECONE_API_KEY:
        pinecone_manager = PineconeManager()
        
    chunker = Chunker(chunk_size=120, chunk_overlap=20)

    # Load Channels
    channels_path = "channels.json"
    if not storage.exists(channels_path):
         # Try local fallback if not found in storage (for dev)
         local_channels = settings.LOCAL_DATA_DIR / "channels.json"
         if local_channels.exists():
             with open(local_channels, 'r') as f:
                 channels = json.load(f)
         else:
             logger.error("channels.json not found.")
             return
    else:
        channels = storage.load_json(channels_path)

    # 채널 필터링 (특정 채널만 처리하거나 전체 처리)
    channels_to_process = channels
    if specific_channel:
        channels_to_process = [c for c in channels if c['channel_id'] == specific_channel]
        if not channels_to_process:
            logger.error(f"Channel not found: {specific_channel}")
            return

    total_channels = len(channels_to_process)
    processed_channels = 0

    for channel_idx, channel in enumerate(channels_to_process, 1):
        channel_id = channel['channel_id']
        channel_name = channel['name']

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"[채널 {channel_idx}/{total_channels}] {channel_name}")
        logger.info(f"채널 ID: {channel_id}")
        logger.info("=" * 60)

        # 이미 완료된 채널 스킵
        if skip_completed_channels and channel_state_manager.is_channel_completed(channel_id):
            logger.info(f"✅ 이미 완료된 채널입니다. 스킵합니다.")
            processed_channels += 1
            continue

        # 1. Fetch Videos
        sort_by = sort_override or settings.VIDEO_SORT_BY
        
        if sort_by == "both":
            logger.info(f"Fetching both 'recent' and 'views' (max {max_results} each)")
            recent_vids = yt_collector.get_channel_videos_sorted(
                channel_id,
                max_results=max_results,
                sort_by="recent"
            )
            popular_vids = yt_collector.get_channel_videos_sorted(
                channel_id,
                max_results=max_results,
                sort_by="views",
                fetch_pool=settings.VIDEO_FETCH_POOL
            )
            # Merge and deduplicate
            video_dict = {v['video_id']: v for v in (recent_vids + popular_vids)}
            videos = list(video_dict.values())
            logger.info(f"Combined videos: {len(recent_vids)} recent + {len(popular_vids)} views -> {len(videos)} unique")
        else:
            videos = yt_collector.get_channel_videos_sorted(
                channel_id,
                max_results=max_results,
                sort_by=sort_by,
                fetch_pool=settings.VIDEO_FETCH_POOL
            )

        if not videos:
            logger.warning(f"채널에서 비디오를 찾을 수 없습니다: {channel_name}")
            continue

        # 채널 상태 초기화
        channel_state_manager.init_channel(channel_id, channel_name, len(videos))
        logger.info(f"📺 총 {len(videos)}개 비디오 발견")

        # Save video list
        storage.save_json(f"raw/videos/{channel_id}/list.json", videos)

        for video_idx, video in enumerate(videos, 1):
            video_id = video['video_id']
            video_title = video['title']

            # Check Status
            status = state_manager.get_video_status(video_id)
            if status:
                video_status = status.get('status')
                if video_status == 'completed':
                    logger.info(f"  [{video_idx}/{len(videos)}] ⏭️ 스킵 (완료됨): {video_title[:30]}...")
                    channel_state_manager.update_video_result(channel_id, success=True, skipped=True)
                    continue
                elif video_status == 'failed' and not retry_failed:
                    logger.info(f"  [{video_idx}/{len(videos)}] ⏭️ 스킵 (실패): {video_title[:30]}...")
                    channel_state_manager.update_video_result(channel_id, success=False, skipped=True)
                    continue

            logger.info(f"  [{video_idx}/{len(videos)}] 🎬 처리 중: {video_title[:40]}...")
            state_manager.update_video_status(video_id, "processing", "transcript_download")
            
            try:
                # 2. Download Transcript (YouTube 차단 방지를 위한 대기)
                time.sleep(3)  # 요청 간 3초 대기
                transcript = transcript_collector.get_transcript(video_id)
                if not transcript:
                    state_manager.update_video_status(video_id, "failed", error="No transcript")
                    channel_state_manager.update_video_result(
                        channel_id, success=False, error_type="No transcript"
                    )
                    continue
                
                storage.save_json(f"transcripts/{video_id}/raw.json", transcript)
                
                # 3. Refine Transcript
                state_manager.update_video_status(video_id, "processing", "refinement")
                raw_text = format_transcript_to_text(transcript)
                refined_text = gemini.refine_transcript(raw_text)
                
                if not refined_text:
                    logger.warning(f"Refinement failed/empty for {video_id}")
                    state_manager.update_video_status(video_id, "failed", error="Refinement failed")
                    channel_state_manager.update_video_result(
                        channel_id, success=False, error_type="Refinement failed"
                    )
                    continue
                    
                storage.save_json(f"transcripts/{video_id}/refined.json", {"text": refined_text})
                
                # 4. Summarize Video (for context)
                state_manager.update_video_status(video_id, "processing", "summarization")
                video_summary = gemini.summarize_video(refined_text)

                if not video_summary:
                    logger.warning(f"Summary generation failed for {video_id}")
                    state_manager.update_video_status(video_id, "failed", error="Summary failed")
                    channel_state_manager.update_video_result(
                        channel_id, success=False, error_type="Summary failed"
                    )
                    continue

                storage.save_json(f"metadata/{video_id}.json", {
                    **video,
                    "summary": video_summary,
                    "processed_at": time.strftime("%Y-%m-%d")
                })

                # 5. Chunking & Contextual Retrieval (타임스탬프 포함)
                state_manager.update_video_status(video_id, "processing", "chunking")

                # 타임스탬프가 있는 청킹 사용
                chunks_with_timestamps = chunker.split_transcript_with_timestamps(transcript)

                chunk_data_list = []
                vectors_to_upsert = []

                for idx, chunk_info in enumerate(chunks_with_timestamps):
                    chunk_text = chunk_info['text']
                    start_time = chunk_info['start_time']
                    end_time = chunk_info['end_time']

                    # Generate Context and Topics (단일 API 호출로 최적화)
                    context, topics = gemini.generate_chunk_context_and_topics(
                        chunk_text, video_summary
                    )

                    # Combine for Embedding
                    final_text_for_embedding = f"{context}\n\n{chunk_text}"

                    # Generate Embedding
                    embedding = gemini.get_embedding(final_text_for_embedding)

                    chunk_meta = {
                        "chunk_index": idx,
                        "text": chunk_text,
                        "context": context,
                        "video_id": video_id,
                        "video_title": video_title,
                        "channel_id": channel_id,
                        "channel_name": channel['name'],
                        "start_time": start_time,
                        "end_time": end_time,
                        "topics": topics
                    }
                    chunk_data_list.append(chunk_meta)

                    if embedding:
                        # 확장된 메타데이터 생성 (모든 필드 포함)
                        metadata = create_chunk_metadata(
                            chunk_text=chunk_text,
                            context=context,
                            video_id=video_id,
                            video_title=video_title,
                            channel_id=channel_id,
                            channel_name=channel['name'],
                            chunk_index=idx,
                            published_at=video.get('published_at', ''),
                            is_verified_professional=channel.get('is_verified_professional', False),
                            specialty=normalize_specialty(channel.get('specialty')),
                            credentials=channel.get('credentials', ''),
                            timestamp_start=format_timestamp(start_time),
                            timestamp_end=format_timestamp(end_time),
                            topics=topics
                        )
                        vector = {
                            "id": f"{video_id}_{idx}",
                            "values": embedding,
                            "metadata": metadata
                        }
                        vectors_to_upsert.append(vector)

                storage.save_json(f"chunks/{video_id}/chunks.json", chunk_data_list)

                # 6. Indexing - 조건부 완료 처리
                if not vectors_to_upsert:
                    logger.warning(f"No embeddings generated for {video_id}")
                    state_manager.update_video_status(video_id, "failed", error="No embeddings generated")
                    channel_state_manager.update_video_result(
                        channel_id, success=False, error_type="No embeddings"
                    )
                    continue

                if pinecone_manager:
                    state_manager.update_video_status(video_id, "processing", "indexing")
                    pinecone_manager.upsert_vectors(vectors_to_upsert)
                    state_manager.update_video_status(video_id, "completed")
                    channel_state_manager.update_video_result(channel_id, success=True)
                    logger.info(f"    ✅ 완료: {video_id}")
                else:
                    state_manager.update_video_status(video_id, "processed_no_index")
                    channel_state_manager.update_video_result(channel_id, success=True)
                    logger.info(f"    ✅ 완료 (인덱싱 제외): {video_id}")

            except Exception as e:
                logger.error(f"    ❌ 에러: {video_id} - {e}")
                state_manager.update_video_status(video_id, "failed", error=str(e))
                channel_state_manager.update_video_result(
                    channel_id, success=False, error_type=str(e)[:50]
                )

        # 채널 처리 완료
        channel_state_manager.complete_channel(channel_id)
        processed_channels += 1

        # 채널 처리 결과 출력
        channel_status = channel_state_manager.get_channel_status(channel_id)
        if channel_status:
            logger.info("")
            logger.info(f"📊 채널 처리 완료: {channel_name}")
            logger.info(f"   - 처리됨: {channel_status.get('processed_videos', 0)}")
            logger.info(f"   - 실패: {channel_status.get('failed_videos', 0)}")
            logger.info(f"   - 자막없음: {channel_status.get('no_transcript_videos', 0)}")
            logger.info(f"   - 스킵: {channel_status.get('skipped_videos', 0)}")

    # 전체 처리 완료 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 전체 파이프라인 처리 완료!")
    logger.info("=" * 60)

    summary = channel_state_manager.get_summary()
    logger.info(f"📊 전체 요약:")
    logger.info(f"   - 처리된 채널: {summary['completed_channels']}/{summary['total_channels']}")
    logger.info(f"   - 총 처리된 비디오: {summary['total_videos_processed']}")
    logger.info(f"   - 총 실패 비디오: {summary['total_videos_failed']}")
    logger.info(f"   - 자막 없는 비디오: {summary['total_no_transcript']}")


def show_status() -> None:
    """채널별 처리 현황을 출력합니다."""
    channel_state_manager = ChannelStateManager()
    summary = channel_state_manager.get_summary()

    print("\n" + "=" * 70)
    print("📊 Medical RAG Pipeline - 채널 처리 현황")
    print("=" * 70)

    print(f"\n🔢 전체 통계:")
    print(f"   - 총 채널 수: {summary['total_channels']}")
    print(f"   - 완료된 채널: {summary['completed_channels']}")
    print(f"   - 처리 중 채널: {summary['processing_channels']}")
    print(f"   - 총 처리된 비디오: {summary['total_videos_processed']}")
    print(f"   - 총 실패 비디오: {summary['total_videos_failed']}")
    print(f"   - 자막 없는 비디오: {summary['total_no_transcript']}")

    if summary['channels']:
        print(f"\n📺 채널별 현황:")
        print("-" * 70)
        print(f"{'채널명':<30} {'상태':<12} {'진행률':<10} {'성공':<6} {'실패':<6} {'스킵':<6}")
        print("-" * 70)

        for ch in summary['channels']:
            status_emoji = "✅" if ch['status'] == 'completed' else "🔄" if ch['status'] == 'processing' else "⏸️"
            name = ch['name'][:28] + ".." if len(ch['name']) > 30 else ch['name']
            print(f"{name:<30} {status_emoji} {ch['status']:<10} {ch['progress']:<10} {ch['processed']:<6} {ch['failed']:<6} {ch['skipped']:<6}")

        print("-" * 70)

    print()


def list_channels() -> None:
    """등록된 채널 목록을 출력합니다."""
    channels_path = settings.LOCAL_DATA_DIR / "channels.json"

    if not channels_path.exists():
        # 루트 디렉토리에서 시도
        channels_path = settings.LOCAL_DATA_DIR.parent / "channels.json"

    if not channels_path.exists():
        print("❌ channels.json 파일을 찾을 수 없습니다.")
        return

    with open(channels_path, 'r', encoding='utf-8') as f:
        channels = json.load(f)

    print("\n" + "=" * 70)
    print("📺 등록된 YouTube 채널 목록")
    print("=" * 70)

    for idx, ch in enumerate(channels, 1):
        print(f"\n[{idx}] {ch['name']}")
        print(f"    채널 ID: {ch['channel_id']}")
        print(f"    전문분야: {', '.join(ch.get('specialty', []))}")
        print(f"    자격: {ch.get('credentials', 'N/A')}")
        print(f"    설명: {ch.get('description', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"총 {len(channels)}개 채널 등록됨")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medical RAG Pipeline - YouTube 의료 콘텐츠 수집 및 인덱싱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                     # 모든 채널 처리 (완료된 채널 스킵)
  python main.py --retry             # 실패한 비디오 재시도
  python main.py --channel UC...     # 특정 채널만 처리
  python main.py --status            # 채널별 처리 현황 확인
  python main.py --list-channels     # 등록된 채널 목록 확인
  python main.py --reset-all         # 모든 채널 상태 리셋 (주의!)
        """
    )

    parser.add_argument(
        '--retry', action='store_true',
        help='실패한 비디오 재시도'
    )
    parser.add_argument(
        '--channel', type=str, default=None,
        help='특정 채널 ID만 처리'
    )
    parser.add_argument(
        '--no-skip', action='store_true',
        help='완료된 채널도 다시 처리'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='채널별 처리 현황 확인'
    )
    parser.add_argument(
        '--list-channels', action='store_true',
        help='등록된 채널 목록 확인'
    )
    parser.add_argument(
        '--reset-all', action='store_true',
        help='모든 채널 상태 리셋 (주의: 재처리 필요)'
    )

    parser.add_argument(
        '--max-results', type=int, default=50,
        help='수집할 비디오 수 (단일 정렬 기준)'
    )
    parser.add_argument(
        '--sort', type=str, default=None,
        choices=['recent', 'views', 'both'],
        help='비디오 정렬 기준 (recent, views, both)'
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.list_channels:
        list_channels()
    elif args.reset_all:
        confirm = input("⚠️ 모든 채널 상태를 리셋하시겠습니까? (yes/no): ")
        if confirm.lower() == 'yes':
            import os
            state_file = settings.LOCAL_DATA_DIR / "channel_state.json"
            if state_file.exists():
                os.remove(state_file)
                print("✅ 채널 상태가 리셋되었습니다.")
            else:
                print("ℹ️ 리셋할 상태 파일이 없습니다.")
        else:
            print("취소되었습니다.")
    else:
        main(
            retry_failed=args.retry,
            skip_completed_channels=not args.no_skip,
            specific_channel=args.channel,
            max_results=args.max_results,
            sort_override=args.sort
        )
