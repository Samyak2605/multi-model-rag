"""Embedding generation for multi-modal content."""

from typing import List, Dict, Optional
import os
import logging

# Try importing OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try importing sentence-transformers as fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text chunks using OpenAI or sentence-transformers fallback."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small", 
        api_key: Optional[str] = None,
        use_local: bool = False
    ):
        """
        Initialize embedder.
        
        Args:
            model: Model name (OpenAI model or sentence-transformers model)
            api_key: OpenAI API key (defaults to env var)
            use_local: If True, use local sentence-transformers instead of OpenAI
        """
        self.model = model
        self.use_local = bool(use_local)  # Ensure it's always a boolean
        self.local_model = None
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        if use_local or not OPENAI_AVAILABLE:
            # Use local sentence-transformers
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ValueError(
                    "sentence-transformers not available. Install with: pip install sentence-transformers"
                )
            # Use a good default model
            if model == "text-embedding-3-small":
                model = "all-MiniLM-L6-v2"  # 384 dimensions, fast and good quality
            self.logger.info(f"Using local embedding model: {model}")
            self.local_model = SentenceTransformer(model)
            self.client = None
        else:
            # Use OpenAI
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fallback to local if no API key
                self.logger.warning("No OpenAI API key found, falling back to local embeddings")
                self.use_local = True
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
                    self.client = None
                else:
                    raise ValueError("No OpenAI API key and sentence-transformers not available")
            else:
                self.client = OpenAI(api_key=api_key)
                self.local_model = None
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if self.use_local or self.local_model is not None:
            # Use local model
            return self.local_model.encode(text, convert_to_numpy=False).tolist()
        
        # Use OpenAI
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_str = str(e).lower()
            # Check for quota/billing errors
            if "quota" in error_str or "insufficient_quota" in error_str or "429" in error_str:
                self.logger.warning(
                    "OpenAI quota exceeded. Falling back to local embeddings. "
                    "Install sentence-transformers: pip install sentence-transformers"
                )
                # Try to fallback to local if available
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.use_local = True
                    self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
                    return self.local_model.encode(text, convert_to_numpy=False).tolist()
                else:
                    raise ValueError(
                        "OpenAI quota exceeded and sentence-transformers not available. "
                        "Install with: pip install sentence-transformers"
                    )
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if self.use_local or self.local_model is not None:
            # Use local model - much faster for batches
            embeddings = self.local_model.encode(texts, convert_to_numpy=False, show_progress_bar=False)
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        
        # Use OpenAI
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                error_str = str(e).lower()
                # Check for quota errors
                if "quota" in error_str or "insufficient_quota" in error_str or "429" in error_str:
                    self.logger.warning("OpenAI quota exceeded. Falling back to local embeddings.")
                    if SENTENCE_TRANSFORMERS_AVAILABLE:
                        self.use_local = True
                        self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
                        # Process remaining texts with local model
                        remaining = texts[i:]
                        local_embeddings = self.local_model.encode(remaining, convert_to_numpy=False, show_progress_bar=False)
                        local_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in local_embeddings]
                        embeddings.extend(local_embeddings)
                        break
                    else:
                        raise ValueError(
                            "OpenAI quota exceeded. Install sentence-transformers: pip install sentence-transformers"
                        )
                self.logger.error(f"Error in batch embedding: {str(e)}")
                # Fallback to individual embedding
                for text in batch:
                    embeddings.append(self.embed_text(text))
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed list of chunks and add embeddings to metadata."""
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        return chunks

