"""Evaluation metrics for retrieval and generation."""

from typing import List, Dict, Optional
import time
from sklearn.metrics import precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates retrieval and generation performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        ground_truth_pages: Optional[List[int]] = None,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict:
        """
        Evaluate retrieval performance.
        
        Args:
            query: Query string
            retrieved_chunks: Retrieved chunks
            ground_truth_pages: List of relevant page numbers (if available)
            k_values: K values for precision@k
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        if not retrieved_chunks:
            return {"error": "No chunks retrieved"}
        
        # Extract page numbers from retrieved chunks
        retrieved_pages = [
            chunk.get("metadata", {}).get("page_number")
            for chunk in retrieved_chunks
        ]
        
        # Calculate precision@k
        if ground_truth_pages:
            for k in k_values:
                top_k_pages = retrieved_pages[:k]
                relevant_in_top_k = len(set(top_k_pages) & set(ground_truth_pages))
                precision_at_k = relevant_in_top_k / k if k > 0 else 0
                metrics[f"precision@{k}"] = precision_at_k
            
            # Recall@k
            for k in k_values:
                top_k_pages = retrieved_pages[:k]
                relevant_in_top_k = len(set(top_k_pages) & set(ground_truth_pages))
                recall_at_k = relevant_in_top_k / len(ground_truth_pages) if ground_truth_pages else 0
                metrics[f"recall@{k}"] = recall_at_k
        
        # Average confidence score
        confidences = [
            chunk.get("confidence_score", 0.0)
            for chunk in retrieved_chunks
        ]
        metrics["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Number of unique pages
        metrics["unique_pages"] = len(set(retrieved_pages))
        metrics["total_chunks"] = len(retrieved_chunks)
        
        return metrics
    
    def evaluate_generation(
        self,
        question: str,
        answer: Dict,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Evaluate answer generation.
        
        Args:
            question: Original question
            answer: Generated answer dictionary
            ground_truth: Ground truth answer (if available)
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "confidence": answer.get("confidence", 0.0),
            "num_sources": len(answer.get("sources", [])),
            "answer_length": len(answer.get("answer", "")),
            "has_sources": len(answer.get("sources", [])) > 0
        }
        
        # Check if answer contains citations
        answer_text = answer.get("answer", "")
        has_citations = "[" in answer_text and "]" in answer_text
        metrics["has_citations"] = has_citations
        
        return metrics
    
    def measure_latency(self, func, *args, **kwargs) -> Dict:
        """
        Measure function execution latency.
        
        Args:
            func: Function to measure
            *args, **kwargs: Function arguments
            
        Returns:
            Dictionary with latency metrics
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return {
            "result": result,
            "latency_ms": (end_time - start_time) * 1000,
            "latency_s": end_time - start_time
        }

