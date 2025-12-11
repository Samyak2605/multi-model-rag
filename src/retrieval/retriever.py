"""Hybrid retrieval system with dense embeddings and BM25."""

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining dense embeddings and BM25."""
    
    def __init__(
        self,
        persist_directory: str = "./data/vector_store",
        collection_name: str = "imf_documents",
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of ChromaDB collection
            dense_weight: Weight for dense embedding similarity
            bm25_weight: Weight for BM25 score
            rrf_k: Reciprocal Rank Fusion constant
        """
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 will be initialized after adding documents
        self.bm25 = None
        self.bm25_corpus = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_documents(self, chunks: List[Dict]):
        """
        Add documents to both vector store and BM25 index.
        
        Args:
            chunks: List of chunk dictionaries with content and embeddings
        """
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"chunk_{idx}"
            ids.append(chunk_id)
            embeddings.append(chunk.get("embedding", []))
            documents.append(chunk["content"])
            
            # Flatten metadata for ChromaDB
            metadata = chunk.get("metadata", {}).copy()
            metadata["chunk_id"] = chunk_id
            metadatas.append(metadata)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Initialize BM25
        self.bm25_corpus = [chunk["content"] for chunk in chunks]
        tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.logger.info(f"Added {len(chunks)} documents to retriever")
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Query string
            query_embedding: Pre-computed query embedding (optional)
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search (True) or dense only (False)
            
        Returns:
            List of retrieved chunks with scores and metadata
        """
        if use_hybrid and self.bm25 is None:
            self.logger.warning("BM25 not initialized, using dense search only")
            use_hybrid = False
        
        # Dense retrieval - use query_embeddings if provided, otherwise query_texts
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2  # Get more for reranking
            )
        else:
            # ChromaDB will need an embedding function, but we'll use query_texts
            # This requires the collection to have been created with an embedding function
            # For now, we'll assume embeddings are provided or collection has embedding function
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2
                )
            except Exception as e:
                self.logger.error(f"Error querying with text, need embedding: {e}")
                # Fallback: return empty if no embedding provided
                if not query_embedding:
                    return []
                raise
        
        dense_results = []
        if results.get("ids") and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                dense_results.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {},
                    "dense_score": 1 - results["distances"][0][i] if results.get("distances") else 0.0  # Convert distance to similarity
                })
        
        if not use_hybrid:
            return dense_results[:top_k] if dense_results else []
        
        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25
        
        # Create BM25 results
        bm25_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                bm25_results.append({
                    "id": f"chunk_{idx}",
                    "content": self.bm25_corpus[idx],
                    "bm25_score": float(score)
                })
        
        # Combine using RRF (Reciprocal Rank Fusion)
        combined_results = self._rrf_fusion(dense_results, bm25_results, top_k)
        
        return combined_results
    
    def _rrf_fusion(
        self,
        dense_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Combine dense and BM25 results using Reciprocal Rank Fusion."""
        # Create score maps
        dense_scores = {}
        bm25_scores = {}
        
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result["id"]
            dense_scores[doc_id] = {
                "rank": rank,
                "score": result.get("dense_score", 0),
                "result": result
            }
        
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            bm25_scores[doc_id] = {
                "rank": rank,
                "score": result.get("bm25_score", 0),
                "result": result
            }
        
        # Get all unique document IDs
        all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_ids:
            dense_rrf = 0
            bm25_rrf = 0
            
            if doc_id in dense_scores:
                rank = dense_scores[doc_id]["rank"]
                dense_rrf = 1 / (self.rrf_k + rank)
            
            if doc_id in bm25_scores:
                rank = bm25_scores[doc_id]["rank"]
                bm25_rrf = 1 / (self.rrf_k + rank)
            
            # Weighted combination
            combined_score = (
                self.dense_weight * dense_rrf +
                self.bm25_weight * bm25_rrf
            )
            
            # Get the result (prefer dense as it has more metadata)
            result = dense_scores.get(doc_id, {}).get("result") or \
                    bm25_scores.get(doc_id, {}).get("result")
            
            if result:
                result["confidence_score"] = combined_score
                result["dense_rrf"] = dense_rrf
                result["bm25_rrf"] = bm25_rrf
                rrf_scores[doc_id] = {
                    "score": combined_score,
                    "result": result
                }
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]
        
        return [item["result"] for item in sorted_results]
    
    def clear(self):
        """Clear all documents from the retriever."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            self.bm25 = None
            self.bm25_corpus = []
        except Exception as e:
            self.logger.error(f"Error clearing retriever: {str(e)}")

