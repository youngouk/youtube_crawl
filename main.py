import json
import time
from datetime import datetime, timezone
from typing import List, Dict
from config.settings import settings
from src.collectors.youtube_collector import YouTubeCollector
from src.collectors.transcript_collector import TranscriptCollector
from src.processors.gemini_processor import GeminiProcessor
from src.processors.chunker import Chunker
from src.storage.r2_storage import get_storage
from src.storage.state_manager import StateManager
from src.vector_db.pinecone_manager import PineconeManager
from src.utils.logger import logger


def create_chunk_metadata(
    chunk_text: str,
    context: str,
    video_id: str,
    video_title: str,
    channel_name: str,
    chunk_index: int,
    published_at: str,
    is_verified_professional: bool,
    specialty: str
) -> dict:
    """
    Pinecone 저장용 청크 메타데이터를 생성합니다.

    Args:
        chunk_text: 청크 텍스트
        context: Contextual Retrieval로 생성된 컨텍스트
        video_id: YouTube 비디오 ID
        video_title: 비디오 제목
        channel_name: 채널 이름
        chunk_index: 청크 인덱스
        published_at: 비디오 게시일 (ISO 8601)
        is_verified_professional: 의료 전문가 인증 여부
        specialty: 전문 분야 (예: 소아과)

    Returns:
        메타데이터 딕셔너리
    """
    return {
        'text': chunk_text,
        'context': context,
        'video_id': video_id,
        'video_title': video_title,
        'channel_name': channel_name,
        'chunk_index': chunk_index,
        'source_type': 'youtube',
        'video_url': f'https://www.youtube.com/watch?v={video_id}',
        'published_at': published_at,
        'is_verified_professional': is_verified_professional,
        'specialty': specialty,
        'processed_at': datetime.now(timezone.utc).isoformat()
    }


def format_transcript_to_text(transcript_list: List[Dict]) -> str:
    """Converts transcript list to a single string."""
    return " ".join([t['text'] for t in transcript_list])

def main():
    logger.info("Starting Medical RAG Pipeline...")
    
    # Initialize components
    storage = get_storage()
    state_manager = StateManager()
    
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
        
    chunker = Chunker(chunk_size=300, chunk_overlap=50)

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

    for channel in channels:
        channel_id = channel['channel_id']
        logger.info(f"Processing channel: {channel['name']} ({channel_id})")
        
        # 1. Fetch Videos
        videos = yt_collector.get_channel_videos(channel_id, max_results=5) # Limit for PoC
        
        # Save video list
        storage.save_json(f"raw/videos/{channel_id}/list.json", videos)
        
        for video in videos:
            video_id = video['video_id']
            video_title = video['title']
            
            # Check Status
            status = state_manager.get_video_status(video_id)
            if status and status.get('status') == 'completed':
                logger.info(f"Skipping already completed video: {video_id}")
                continue
                
            logger.info(f"Processing video: {video_title} ({video_id})")
            state_manager.update_video_status(video_id, "processing", "transcript_download")
            
            try:
                # 2. Download Transcript
                transcript = transcript_collector.get_transcript(video_id)
                if not transcript:
                    state_manager.update_video_status(video_id, "failed", error="No transcript")
                    continue
                
                storage.save_json(f"transcripts/{video_id}/raw.json", transcript)
                
                # 3. Refine Transcript
                state_manager.update_video_status(video_id, "processing", "refinement")
                raw_text = format_transcript_to_text(transcript)
                refined_text = gemini.refine_transcript(raw_text)
                
                if not refined_text:
                    logger.warning(f"Refinement failed/empty for {video_id}")
                    # Fallback to raw text if refinement fails or is empty?
                    # For now, let's strictly require refinement for quality
                    state_manager.update_video_status(video_id, "failed", error="Refinement failed")
                    continue
                    
                storage.save_json(f"transcripts/{video_id}/refined.json", {"text": refined_text})
                
                # 4. Summarize Video (for context)
                video_summary = gemini.summarize_video(refined_text)
                storage.save_json(f"metadata/{video_id}.json", {
                    **video,
                    "summary": video_summary,
                    "processed_at": time.strftime("%Y-%m-%d")
                })

                # 5. Chunking & Contextual Retrieval
                state_manager.update_video_status(video_id, "processing", "chunking")
                chunks = chunker.split_text(refined_text)
                
                chunk_data_list = []
                vectors_to_upsert = []
                
                for idx, chunk_text in enumerate(chunks):
                    # Generate Context
                    context = gemini.generate_chunk_context(chunk_text, video_summary)
                    
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
                        "channel_name": channel['name']
                    }
                    chunk_data_list.append(chunk_meta)
                    
                    if embedding:
                        # 확장된 메타데이터 생성
                        metadata = create_chunk_metadata(
                            chunk_text=chunk_text,
                            context=context,
                            video_id=video_id,
                            video_title=video_title,
                            channel_name=channel['name'],
                            chunk_index=idx,
                            published_at=video.get('published_at', ''),
                            is_verified_professional=channel.get('is_verified_professional', False),
                            specialty=channel.get('specialty', '')
                        )
                        vector = {
                            "id": f"{video_id}_{idx}",
                            "values": embedding,
                            "metadata": metadata
                        }
                        vectors_to_upsert.append(vector)
                    
                storage.save_json(f"chunks/{video_id}/chunks.json", chunk_data_list)
                
                # 6. Indexing
                if pinecone_manager and vectors_to_upsert:
                    state_manager.update_video_status(video_id, "processing", "indexing")
                    pinecone_manager.upsert_vectors(vectors_to_upsert)
                
                state_manager.update_video_status(video_id, "completed")
                logger.info(f"Completed video: {video_id}")
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                state_manager.update_video_status(video_id, "failed", error=str(e))

if __name__ == "__main__":
    main()
