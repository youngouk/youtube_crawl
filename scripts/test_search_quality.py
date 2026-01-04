#!/usr/bin/env python3
"""
벡터 검색 품질 테스트 스크립트.

10가지 다양한 의료 관련 쿼리로 검색 품질을 검증합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.gemini_processor import GeminiProcessor
from src.vector_db.pinecone_manager import PineconeManager


def run_search_tests():
    """10가지 검색 테스트 실행."""
    print("=" * 70)
    print("🔍 벡터 검색 품질 테스트")
    print("=" * 70)
    print()

    # 초기화
    print("🔧 초기화 중...")
    processor = GeminiProcessor()
    pinecone = PineconeManager()

    stats = pinecone.get_index_stats()
    print(f"  - 인덱스: {pinecone.index_name}")
    print(f"  - 총 벡터: {stats.get('total_vector_count', 0)}개")
    print(f"  - 차원: {stats.get('dimension', 0)}")
    print()

    # 10가지 테스트 쿼리 (다양한 의료 주제)
    test_queries = [
        {
            "id": 1,
            "query": "아기 이유식 시작 시기와 방법",
            "category": "소아과/육아",
            "expected_topics": ["이유식", "영유아", "육아"]
        },
        {
            "id": 2,
            "query": "고혈압 약 복용 시 주의사항",
            "category": "내과/심혈관",
            "expected_topics": ["고혈압", "약물", "혈압"]
        },
        {
            "id": 3,
            "query": "당뇨병 환자 식이요법",
            "category": "내과/당뇨",
            "expected_topics": ["당뇨", "혈당", "식이"]
        },
        {
            "id": 4,
            "query": "허리 디스크 증상과 치료",
            "category": "정형외과",
            "expected_topics": ["디스크", "허리", "통증"]
        },
        {
            "id": 5,
            "query": "피부 아토피 관리 방법",
            "category": "피부과",
            "expected_topics": ["아토피", "피부", "보습"]
        },
        {
            "id": 6,
            "query": "수면 장애 불면증 해결",
            "category": "정신과/신경과",
            "expected_topics": ["수면", "불면", "수면위생"]
        },
        {
            "id": 7,
            "query": "임신 중 영양제 섭취",
            "category": "산부인과",
            "expected_topics": ["임신", "영양", "엽산"]
        },
        {
            "id": 8,
            "query": "콜레스테롤 수치 관리",
            "category": "내과/대사",
            "expected_topics": ["콜레스테롤", "지질", "심혈관"]
        },
        {
            "id": 9,
            "query": "위염 위산 역류 증상",
            "category": "소화기내과",
            "expected_topics": ["위염", "역류", "소화"]
        },
        {
            "id": 10,
            "query": "예방접종 스케줄 부작용",
            "category": "소아과/예방의학",
            "expected_topics": ["예방접종", "백신", "면역"]
        }
    ]

    print("=" * 70)
    print("📋 검색 테스트 시작")
    print("=" * 70)

    results_summary = []

    for test in test_queries:
        print(f"\n[테스트 {test['id']}/10] {test['category']}")
        print(f"  쿼리: \"{test['query']}\"")
        print("-" * 50)

        # 임베딩 생성
        embedding = processor.get_embedding(test['query'])
        if not embedding:
            print("  ❌ 임베딩 생성 실패")
            results_summary.append({
                "id": test['id'],
                "success": False,
                "error": "임베딩 생성 실패"
            })
            continue

        print(f"  ✅ 임베딩 생성 완료 (차원: {len(embedding)})")

        # 검색 실행
        try:
            results = pinecone.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )

            if not results:
                print("  ⚠️ 검색 결과 없음")
                results_summary.append({
                    "id": test['id'],
                    "success": False,
                    "error": "검색 결과 없음"
                })
                continue

            print(f"  📊 상위 3개 결과:")

            scores = []
            for i, result in enumerate(results[:3], 1):
                score = result.get('score', 0)
                scores.append(score)
                metadata = result.get('metadata', {})

                video_title = metadata.get('video_title', 'N/A')[:40]
                channel = metadata.get('channel_name', 'N/A')[:20]
                chunk_text = metadata.get('chunk_text', '')[:80]

                print(f"    {i}. 유사도: {score:.4f}")
                print(f"       제목: {video_title}...")
                print(f"       채널: {channel}")
                print(f"       내용: {chunk_text}...")
                print()

            avg_score = sum(scores) / len(scores) if scores else 0

            # 품질 판정 (유사도 0.3 이상이면 양호)
            quality = "✅ 양호" if avg_score >= 0.3 else "⚠️ 검토필요" if avg_score >= 0.2 else "❌ 저품질"

            results_summary.append({
                "id": test['id'],
                "success": True,
                "avg_score": avg_score,
                "top_score": max(scores) if scores else 0,
                "quality": quality
            })

            print(f"  📈 평균 유사도: {avg_score:.4f} {quality}")

        except Exception as e:
            print(f"  ❌ 검색 실패: {e}")
            results_summary.append({
                "id": test['id'],
                "success": False,
                "error": str(e)
            })

    # 최종 요약
    print()
    print("=" * 70)
    print("📊 최종 테스트 결과 요약")
    print("=" * 70)

    success_count = sum(1 for r in results_summary if r.get('success'))
    avg_scores = [r['avg_score'] for r in results_summary if r.get('avg_score')]

    print(f"\n  테스트 성공: {success_count}/10")

    if avg_scores:
        overall_avg = sum(avg_scores) / len(avg_scores)
        print(f"  전체 평균 유사도: {overall_avg:.4f}")
        print(f"  최고 유사도: {max(avg_scores):.4f}")
        print(f"  최저 유사도: {min(avg_scores):.4f}")

    print("\n  개별 결과:")
    for r in results_summary:
        if r.get('success'):
            print(f"    테스트 {r['id']}: {r['quality']} (평균: {r['avg_score']:.4f}, 최고: {r['top_score']:.4f})")
        else:
            print(f"    테스트 {r['id']}: ❌ 실패 - {r.get('error', '알 수 없음')}")

    # 최종 판정
    print()
    if success_count == 10 and overall_avg >= 0.3:
        print("🎉 검색 품질 검증 완료: 모든 테스트 통과!")
    elif success_count >= 8 and overall_avg >= 0.25:
        print("✅ 검색 품질 양호: 대부분의 테스트 통과")
    else:
        print("⚠️ 검색 품질 개선 필요")

    print()


if __name__ == "__main__":
    run_search_tests()
