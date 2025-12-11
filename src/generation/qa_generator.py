"""Question-Answering generation with source attribution."""

from typing import List, Dict, Optional
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates answers with source citations."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize QA generator.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
        self.system_prompt = """You are a financial document analysis assistant. 
Answer questions based on the provided context from IMF reports and financial documents.
- Always cite your sources using [Page X] or [Section Y] format
- If information is not in the context, clearly state that
- Be precise and factual
- For numerical data, cite the exact source"""
    
    def generate(
        self,
        question: str,
        context_chunks: List[Dict],
        include_sources: bool = True
    ) -> Dict:
        """
        Generate answer from question and context.
        
        Args:
            question: User question
            context_chunks: Retrieved context chunks
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not context_chunks:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context string
        context_parts = []
        sources = []
        
        for chunk in context_chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            page_num = metadata.get("page_number", "Unknown")
            source = metadata.get("source", f"Page {page_num}")
            
            context_parts.append(f"[Source: {source}]\n{content}")
            sources.append({
                "page": page_num,
                "source": source,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "confidence": chunk.get("confidence_score", 0.0)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        user_prompt = f"""Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above. Include source citations in [Page X] format."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Calculate average confidence
            avg_confidence = sum(s["confidence"] for s in sources) / len(sources) if sources else 0.0
            
            return {
                "answer": answer,
                "sources": sources if include_sources else [],
                "confidence": avg_confidence,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def generate_multi_hop(
        self,
        question: str,
        retriever,
        embedder=None,
        max_hops: int = 2
    ) -> Dict:
        """
        Multi-hop reasoning: retrieve, generate, then retrieve again if needed.
        
        Args:
            question: User question
            retriever: HybridRetriever instance
            embedder: Embedder instance for query embeddings
            max_hops: Maximum retrieval hops
            
        Returns:
            Dictionary with answer and reasoning chain
        """
        all_chunks = []
        reasoning_steps = []
        
        # Initial retrieval
        if embedder:
            query_embedding = embedder.embed_text(question)
            chunks = retriever.retrieve(question, query_embedding=query_embedding, top_k=5)
        else:
            # Fallback: try without embedding (may fail)
            try:
                chunks = retriever.retrieve(question, top_k=5, use_hybrid=False)
            except:
                chunks = []
        
        all_chunks.extend(chunks)
        reasoning_steps.append({
            "step": 1,
            "query": question,
            "chunks_retrieved": len(chunks)
        })
        
        # Generate intermediate answer if needed
        if max_hops > 1 and chunks:
            intermediate_answer = self.generate(question, chunks, include_sources=False)
            
            # Extract key entities/concepts for follow-up retrieval
            follow_up_query = f"{question} {intermediate_answer['answer'][:200]}"
            
            if embedder:
                follow_up_embedding = embedder.embed_text(follow_up_query)
                follow_up_chunks = retriever.retrieve(follow_up_query, query_embedding=follow_up_embedding, top_k=3)
            else:
                try:
                    follow_up_chunks = retriever.retrieve(follow_up_query, top_k=3, use_hybrid=False)
                except:
                    follow_up_chunks = []
            
            # Merge unique chunks
            existing_ids = {c.get("id") for c in all_chunks}
            for chunk in follow_up_chunks:
                if chunk.get("id") not in existing_ids:
                    all_chunks.append(chunk)
            
            reasoning_steps.append({
                "step": 2,
                "query": follow_up_query,
                "chunks_retrieved": len(follow_up_chunks)
            })
        
        # Final answer
        final_answer = self.generate(question, all_chunks[:5], include_sources=True)
        final_answer["reasoning_steps"] = reasoning_steps
        
        return final_answer

