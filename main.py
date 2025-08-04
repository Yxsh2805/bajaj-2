class LocalEmbeddingVectorStore:
    def __init__(self):
        # Download model on first use (not during Docker build)
        logger.info("Loading local embedding model (downloading if needed)...")
        try:
            # Use a smaller, faster model that downloads quicker
            self.model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, downloads in ~10-15 seconds
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to an even smaller model if the first one fails
            logger.info("Trying fallback model...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        
        self.documents = []
        self.vectors = []
    
    def add_documents_local_batch(self, documents: List[Document]):
        """ULTRA-FAST: Process ALL documents in ONE batch call"""
        logger.info(f"LOCAL BATCH: Processing {len(documents)} chunks in ONE call")
        
        start_time = time.time()
        
        # Extract all text content
        texts = [doc.page_content for doc in documents]
        
        # SINGLE batch embedding call - NO individual API calls!
        logger.info("Generating embeddings with local model...")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=16,  # Reduced batch size for Railway memory limits
                show_progress_bar=False,
                convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            # Fallback: process in smaller batches
            embeddings = []
            for i in range(0, len(texts), 8):  # Very small batches
                batch_texts = texts[i:i + 8]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            embeddings = np.array(embeddings)
        
        # Store results
        self.documents = documents
        self.vectors = embeddings
        
        embedding_time = time.time() - start_time
        logger.info(f"LOCAL BATCH: Completed in {embedding_time:.2f}s ({len(documents)} chunks)")
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Fast local similarity search"""
        if len(self.vectors) == 0:
            return []
        
        try:
            # Single query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Fast similarity calculation
            similarities = cosine_similarity(query_embedding, self.vectors)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            return [self.documents[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            # Return first k documents as fallback
            return self.documents[:k] if len(self.documents) >= k else self.documents
