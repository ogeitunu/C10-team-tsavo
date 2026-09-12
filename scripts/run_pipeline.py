import os
import sys
import pandas as pd

# Append root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import enrich_documents
from src.bm25_indexer import LexicalBM25Retriever
from src.dense_encoder import DenseRetriever
from src.rrf_fusion import reciprocal_rank_fusion
from src.dual_reranker import DualCrossEncoderReranker

def main():
    print("🌾 Starting Team Tsavo RAG Submission Pipeline...")
    
    # Paths
    docs_path = "data/documents.csv"
    test_queries_path = "data/test_queries.csv"
    output_path = "submission.csv"

    # Load Data
    print("1. Loading and Enriching Corpus...")
    docs_df = pd.read_csv(docs_path)
    test_df = pd.read_csv(test_queries_path)
    enriched_docs_df = enrich_documents(docs_df)
    
    doc_text_map = dict(zip(enriched_docs_df['document_id'].astype(str), enriched_docs_df['enriched_text']))

    # Initialize Stage 1 Retrievers
    print("2. Fitting Sparse (BM25) and Encoding Dense (BGE) Vectors...")
    bm25_retriever = LexicalBM25Retriever()
    bm25_retriever.fit(enriched_docs_df)

    dense_retriever = DenseRetriever()
    dense_retriever.encode_documents(enriched_docs_df)

    # Initialize Stage 2 Dual Re-Ranker
    print("3. Loading Dual Cross-Encoder Ensemble...")
    reranker = DualCrossEncoderReranker()

    submission_rows = []
    print("4. Executing Retrieval and Re-Ranking over Test Queries...")
    
    for idx, row in test_df.iterrows():
        query_id = str(row['query_id'])
        query = str(row['query'])
        
        # Stage 1 Retrieval
        bm25_hits = bm25_retriever.retrieve(query, top_k=30)
        dense_hits = dense_retriever.retrieve(query, top_k=30)
        
        # RRF Fusion
        candidate_ids = reciprocal_rank_fusion(bm25_hits, dense_hits, k=15, top_n=20)
        candidates = [(doc_id, doc_text_map[doc_id]) for doc_id in candidate_ids if doc_id in doc_text_map]
        
        # Stage 2 Dual Re-Ranking
        top_5_results = reranker.rerank(query, candidates, top_k=5)
        
        # Build CSV Rows (QueryId, DocumentId)
        for doc_id, _ in top_5_results:
            submission_rows.append({"QueryId": query_id, "DocumentId": doc_id})

    # Export Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_path, index=False)
    print(f"✅ Pipeline Completed! Exported {len(sub_df)} rows to {output_path}")

if __name__ == "__main__":
    main()
