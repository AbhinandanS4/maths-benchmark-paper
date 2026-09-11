import re
import numpy as np
from rank_bm25 import BM25Okapi
from rag_dense import DenseRAG

def math_tokenizer(text: str) -> list[str]:
    """Preserves math variables and numbers rather than stripping them like standard NLP tokenizers."""
    # Keeps alpha-numeric terms, variables with sub/superscripts (a_n, S_20), and math symbols
    return re.findall(r"\b\w+_\w+|\b\w+\b|[+\-*/=]", text.lower())

class HybridRAG:
    def __init__(self, dense_rag: DenseRAG):
        self.dense_rag = dense_rag
        
        # Pull indexed documents directly from Chroma collection
        all_data = self.dense_rag.collection.get()
        self.documents = all_data.get("documents", [])
        self.doc_ids = all_data.get("ids", [])
        
        if not self.documents:
            print("[HybridRAG] Warning: Chroma store is empty. Hybrid search unavailable.")
            self.bm25 = None
            return

        # Build in-memory BM25 index on the extracted corpus
        tokenized_corpus = [math_tokenizer(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[HybridRAG] BM25 index initialized over {len(self.documents)} documents.")

    def _reciprocal_rank_fusion(
        self, 
        sparse_ranks: list[int], 
        dense_ranks: list[int], 
        rrf_k: int = 60
    ) -> list[int]:
        """Merges sparse and dense document indices via RRF scoring."""
        rrf_scores = {}

        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort indices by descending merged RRF score
        fused_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        return fused_indices

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.bm25 or not self.documents:
            return "No index available for retrieval."

        # 1. Sparse Search (BM25)
        query_tokens = math_tokenizer(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        sparse_ranked_indices = np.argsort(bm25_scores)[::-1].tolist()

        # 2. Dense Search (Chroma)
        dense_raw = self.dense_rag.collection.query(
            query_texts=[query],
            n_results=len(self.documents)  # Pull full rankings to perform proper RRF fusion
        )
        dense_ids = dense_raw.get("ids", [[]])[0]
        
        # Map string IDs back to integer indices
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        dense_ranked_indices = [id_to_idx[doc_id] for doc_id in dense_ids if doc_id in id_to_idx]

        # 3. Fuse Rankings via RRF
        fused_indices = self._reciprocal_rank_fusion(sparse_ranked_indices, dense_ranked_indices)
        
        # Return top K context excerpts
        retrieved_chunks = [self.documents[idx] for idx in fused_indices[:top_k]]
        return "\n\n---\n\n".join(retrieved_chunks)