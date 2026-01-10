#!/usr/bin/env python3
"""
Pinecone 데이터 재구축 스크립트

R2에서 원본 데이터를 읽어 새로운 스키마로 Pinecone에 재업로드합니다.
- 청킹: 120토큰, 오버랩 20
- ID 형식: {video_id}_chunk_{n}
- 필드명: chunk_text, start_time, end_time (구버전)
- 메타데이터: specialty, credentials 등 (신버전 추가)

사용법:
    python scripts/rebuild_pinecone.py              # 전체 실행
    python scripts/rebuild_pinecone.py --resume     # 중단 지점부터 재개
    python scripts/rebuild_pinecone.py --dry-run    # 테스트 (실제 저장 안함)
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

# 외부 라이브러리 (실제 실행 시 필요)
# boto3, pinecone, tiktoken은 Task 2에서 사용
try:
    import boto3
    from botocore.config import Config
    from pinecone import Pinecone
    import tiktoken
except ImportError as e:
    # --help 실행 시에는 필요 없음
    pass

# 프로젝트 루트를 path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 프로젝트 모듈 (실제 실행 시 필요)
try:
    from config.settings import settings
    from src.processors.gemini_processor import GeminiProcessor
except ImportError:
    # --help 실행 시에는 필요 없음
    pass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 상수
CHUNK_SIZE = 120          # 토큰
CHUNK_OVERLAP = 20        # 토큰
PINECONE_BATCH_SIZE = 100
PROGRESS_FILE = "data/rebuild_progress.json"


def log(msg: str = "") -> None:
    """실시간 로그 출력"""
    print(msg, flush=True)


class ProgressTracker:
    """진행 상태 추적 (중단/재개 지원)"""

    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = Path(progress_file)
        self.data = self._load()

    def _load(self) -> dict:
        """진행 상태 파일 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed_videos": [],
            "failed_videos": [],
            "total_chunks": 0,
            "started_at": None,
            "last_updated": None
        }

    def save(self) -> None:
        """진행 상태 저장"""
        self.data["last_updated"] = datetime.now().isoformat()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def is_completed(self, video_id: str) -> bool:
        """비디오가 이미 처리되었는지 확인"""
        return video_id in self.data["completed_videos"]

    def mark_completed(self, video_id: str, chunk_count: int) -> None:
        """비디오 처리 완료 표시"""
        if video_id not in self.data["completed_videos"]:
            self.data["completed_videos"].append(video_id)
            self.data["total_chunks"] += chunk_count
        self.save()

    def mark_failed(self, video_id: str, error: str) -> None:
        """비디오 처리 실패 표시"""
        self.data["failed_videos"].append({"video_id": video_id, "error": error})
        self.save()

    def start(self) -> None:
        """작업 시작 시간 기록"""
        if not self.data["started_at"]:
            self.data["started_at"] = datetime.now().isoformat()
        self.save()

    def reset(self) -> None:
        """진행 상태 초기화"""
        self.data = {
            "completed_videos": [],
            "failed_videos": [],
            "total_chunks": 0,
            "started_at": datetime.now().isoformat(),
            "last_updated": None
        }
        self.save()

    def get_summary(self) -> dict:
        """진행 상태 요약 반환"""
        return {
            "completed_count": len(self.data["completed_videos"]),
            "failed_count": len(self.data["failed_videos"]),
            "total_chunks": self.data["total_chunks"],
            "started_at": self.data["started_at"],
            "last_updated": self.data["last_updated"]
        }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Pinecone 데이터 재구축')
    parser.add_argument('--resume', action='store_true', help='중단 지점부터 재개')
    parser.add_argument('--dry-run', action='store_true', help='테스트 모드 (실제 저장 안함)')
    parser.add_argument('--limit', type=int, default=0, help='처리할 비디오 수 제한 (0=전체)')
    args = parser.parse_args()

    log("=" * 60)
    log("Pinecone 데이터 재구축 스크립트")
    log("=" * 60)
    log(f"청크 크기: {CHUNK_SIZE}토큰, 오버랩: {CHUNK_OVERLAP}토큰")
    log(f"모드: {'DRY-RUN (테스트)' if args.dry_run else '실제 실행'}")
    log(f"재개 모드: {'예' if args.resume else '아니오'}")
    if args.limit > 0:
        log(f"처리 제한: {args.limit}개 비디오")
    log()

    # TODO: 구현 예정 (Task 2부터)
    log("⚠️ 아직 구현되지 않았습니다. Task 2부터 구현 필요.")
    log()
    log("예정 기능:")
    log("  1. R2에서 원본 트랜스크립트 로드")
    log("  2. 채널 메타데이터(specialty, credentials) 매핑")
    log("  3. 120토큰 청킹 (오버랩 20토큰)")
    log("  4. Gemini 임베딩 생성")
    log("  5. Pinecone 업서트 (구버전 필드명 사용)")
    log()

    # 진행 상태 추적기 테스트
    tracker = ProgressTracker()
    summary = tracker.get_summary()
    log(f"현재 진행 상태: {summary['completed_count']}개 완료, {summary['total_chunks']}개 청크")


if __name__ == "__main__":
    main()
