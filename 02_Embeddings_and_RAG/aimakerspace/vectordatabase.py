import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.metadata = {}  # Store metadata for each document
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with optional metadata into the database."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for vectors similar to the query vector.
        Optionally filter results based on metadata criteria.
        """
        scores = []
        for key, vector in self.vectors.items():
            # Apply filter if specified
            if filter_criteria and key in self.metadata:
                # Check if document meets all filter criteria
                meets_criteria = all(
                    key in self.metadata and 
                    field in self.metadata[key] and 
                    self.metadata[key][field] == value
                    for field, value in filter_criteria.items()
                )
                if not meets_criteria:
                    continue
                    
            similarity = distance_measure(query_vector, vector)
            scores.append((key, similarity))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search by text query with optional metadata filtering.
        Returns either text-similarity pairs or just text depending on return_as_text.
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filter_criteria)
        
        if return_as_text:
            return [result[0] for result in results]
        
        return results

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document if it exists."""
        return self.metadata.get(key)
    
    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(
        self, 
        list_of_text: List[str], 
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorDatabase":
        """
        Build vector database from a list of texts with optional metadata.
        If metadata_list is provided, it should be the same length as list_of_text.
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        
        if metadata_list and len(metadata_list) != len(list_of_text):
            raise ValueError("metadata_list must be the same length as list_of_text")
            
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list else None
            self.insert(text, np.array(embedding), metadata)
            
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    # Example metadata
    metadata_list = [
        {"source": "food_notes.txt", "type": "text", "topic": "food"},
        {"source": "food_notes.txt", "type": "text", "topic": "food"},
        {"source": "pet_notes.txt", "type": "text", "topic": "pets"},
        {"source": "pet_notes.txt", "type": "text", "topic": "pets"},
        {"source": "pet_notes.txt", "type": "text", "topic": "pets"},
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text, metadata_list))
    k = 2

    # Search without filtering
    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    # Search with filtering
    filtered_results = vector_db.search_by_text(
        "I think fruit is awesome!", 
        k=k, 
        filter_criteria={"topic": "food"}
    )
    print(f"Filtered to food topics ({k}):", filtered_results)

    # Get metadata for a document
    retrieved_metadata = vector_db.get_metadata("I like to eat broccoli and bananas.")
    print("Retrieved metadata:", retrieved_metadata)

    # Retrieving text instead of (text, similarity) pairs
    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
