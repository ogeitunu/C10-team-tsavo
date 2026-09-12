from typing import List, Tuple, Dict

def reciprocal_rank_fusion(
    bm25_results: List[Tuple[str, float]], 
    dense_results: List[Tuple[str, float]], 
    k: int = 15, 
    top_n: int = 20
) -> List[str]:
    """
    Combines sparse (BM25) and dense retrieval rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score = sum(1 / (k + rank))
    """
    rrf_scores: Dict[str, float] = {}

    # Accumulate BM25 ranks
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Accumulate Dense ranks
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort documents by total fused score
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_n]]
