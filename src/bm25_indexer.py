import re
import pandas as pd
from rank_bm25 import BM25Okapi

class LexicalBM25Retriever:
    """Sparse Lexical Retriever using BM25 with custom tokenization for agronomic terms."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []

    def _tokenize(self, text: str):
        # Retains alphanumeric tokens, NPK ratios (e.g., 15-15-15), and hyphens
        return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())

    def fit(self, docs_df: pd.DataFrame, text_col: str = 'enriched_text', id_col: str = 'document_id'):
        self.doc_ids = docs_df[id_col].astype(str).tolist()
        corpus = docs_df[text_col].astype(str).tolist()
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 30):
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort scores in descending order
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]
