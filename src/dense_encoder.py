import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """Dense Vector Search utilizing BAAI/bge-large-en-v1.5 with asymmetric instruction prompts."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.doc_embeddings = None
        self.doc_ids = []

    def encode_documents(self, docs_df: pd.DataFrame, text_col: str = 'enriched_text', id_col: str = 'document_id'):
        self.doc_ids = docs_df[id_col].astype(str).tolist()
        passages = docs_df[text_col].astype(str).tolist()
        
        # Direct passage encoding (No HyDE preamble to prevent noise contamination)
        self.doc_embeddings = self.model.encode(
            passages, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 30):
        # Asymmetric instruction prompt for BGE queries
        instruction_query = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.model.encode(
            instruction_query, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        # Cosine similarity matrix dot-product
        similarity_scores = torch.mm(query_embedding.unsqueeze(0), self.doc_embeddings.T).squeeze(0)
        top_scores, top_indices = torch.topk(similarity_scores, k=min(top_k, len(self.doc_ids)))
        
        return [(self.doc_ids[idx.item()], score.item()) for score, idx in zip(top_scores, top_indices)]
