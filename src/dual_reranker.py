import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DualCrossEncoderReranker:
    """Ensemble Cross-Encoder Re-Ranking: BGE-Reranker-Large (55%) + MXBAI-Rerank-Large (45%)."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. BGE Reranker
        self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.bge_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large").to(self.device).eval()
        
        # 2. MXBAI Reranker
        self.mxbai_tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-rerank-large")
        self.mxbai_model = AutoModelForSequenceClassification.from_pretrained("mixedbread-ai/mxbai-rerank-large").to(self.device).eval()

    def _score_pairs(self, model, tokenizer, pairs: List[Tuple[str, str]]) -> torch.Tensor:
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            logits = model(**inputs).logits
            if logits.shape[1] == 1:
                scores = logits.squeeze(-1)
            else:
                scores = logits[:, 1]
            return torch.sigmoid(scores) # Sigmoid calibration

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        candidates: List of tuples (doc_id, enriched_text)
        Returns top_k ranked doc_ids.
        """
        if not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        
        # Compute calibrated scores
        bge_scores = self._score_pairs(self.bge_model, self.bge_tokenizer, pairs)
        mxbai_scores = self._score_pairs(self.mxbai_model, self.mxbai_tokenizer, pairs)
        
        # 55% / 45% Weighted Logit Fusion
        fused_scores = 0.55 * bge_scores + 0.45 * mxbai_scores
        
        # Pair document IDs with final ensemble score
        scored_docs = [(candidates[i][0], fused_scores[i].item()) for i in range(len(candidates))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Explicit FP16 memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return scored_docs[:top_k]
