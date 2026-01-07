"""
End-to-End RAG Pipeline for BioMoQA-Ragnarok

Implements the full Ragnarok framework:
1. (R) Retrieval: SIBILS API
2. Reranking (optional): LLM-based reranking
3. (AG) Augmented Generation: Open-source LLM with citations
"""

from typing import List, Dict, Optional
import time
from dataclasses import dataclass

from .retrieval import SIBILSRetriever, Document
from .generation import LLMGenerator


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval
    retrieval_n: int = 100
    retrieval_collection: str = "pmc"

    # Reranking
    rerank: bool = False
    rerank_n: int = 20

    # Generation
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7

    # General
    device: str = "cuda"


class RAGPipeline:
    """
    Complete RAG pipeline for biomedical question answering.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.run("What is the host of Plasmodium falciparum?")
        print(result['answer'])
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or RAGConfig()

        print("Initializing BioMoQA-Ragnarok Pipeline...")
        print(f"Config: {self.config}\n")

        # Initialize retriever
        print("Loading retriever...")
        self.retriever = SIBILSRetriever(
            collection=self.config.retrieval_collection,
            default_n=self.config.retrieval_n,
        )

        # Initialize generator (lazy loading - only when needed)
        self.generator = None

        print("Pipeline initialized successfully!\n")

    def _load_generator(self):
        """Lazy load the LLM generator (only when first needed)."""
        if self.generator is None:
            print("Loading LLM generator (this may take a few minutes)...")
            self.generator = LLMGenerator(
                model_name=self.config.model_name,
                device=self.config.device,
                load_in_4bit=self.config.load_in_4bit,
                max_new_tokens=self.config.max_new_tokens,
            )

    def retrieve(self, question: str) -> List[Document]:
        """
        Retrieval stage (R).

        Args:
            question: User question

        Returns:
            List of retrieved documents
        """
        docs = self.retriever.retrieve(
            question,
            n=self.config.retrieval_n,
        )
        return docs

    def rerank(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Reranking stage (optional).

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Reranked documents (top-k)
        """
        # TODO: Implement proper reranking with cross-encoder or LLM
        # For now, just return top-k by score
        if self.config.rerank:
            return documents[:self.config.rerank_n]
        return documents

    def generate(
        self,
        question: str,
        documents: List[Document],
        topic_id: Optional[str] = None,
    ) -> Dict:
        """
        Augmented generation stage (AG).

        Args:
            question: User question
            documents: Retrieved/reranked documents
            topic_id: Optional topic identifier

        Returns:
            Ragnarok-format output with answer and citations
        """
        # Lazy load generator
        self._load_generator()

        # Format documents for generator
        doc_dicts = [
            {
                "title": doc.title,
                "text": doc.get_text(max_length=1000),  # Limit context length
                "id": doc.doc_id,
            }
            for doc in documents
        ]

        # Generate answer
        output = self.generator.generate_ragnarok_output(
            question=question,
            documents=doc_dicts,
            topic_id=topic_id,
        )

        return output

    def run(
        self,
        question: str,
        topic_id: Optional[str] = None,
        return_documents: bool = False,
    ) -> Dict:
        """
        Run complete RAG pipeline.

        Args:
            question: User question
            topic_id: Optional topic identifier
            return_documents: Include retrieved documents in output

        Returns:
            Complete pipeline output in Ragnarok format
        """
        start_time = time.time()

        print(f"Processing question: {question}\n")

        # Stage 1: Retrieval
        print(f"[1/3] Retrieving documents (n={self.config.retrieval_n})...")
        documents = self.retrieve(question)
        print(f"      Retrieved {len(documents)} documents")

        # Stage 2: Reranking (optional)
        if self.config.rerank:
            print(f"[2/3] Reranking to top-{self.config.rerank_n}...")
            documents = self.rerank(question, documents)
            print(f"      Kept {len(documents)} documents")
        else:
            print(f"[2/3] Skipping reranking (using top-{self.config.rerank_n})...")
            documents = documents[:self.config.rerank_n]

        # Stage 3: Generation
        print(f"[3/3] Generating answer with {self.config.model_name}...")
        output = self.generate(question, documents, topic_id)

        # Add metadata
        output["pipeline_time"] = round(time.time() - start_time, 2)
        output["num_retrieved"] = len(documents)

        if return_documents:
            output["documents"] = [
                {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "score": doc.score,
                }
                for doc in documents
            ]

        print(f"\nCompleted in {output['pipeline_time']:.2f}s\n")

        return output


def main():
    """Example usage of complete RAG pipeline."""
    import json

    # Create pipeline
    config = RAGConfig(
        retrieval_n=50,
        rerank_n=10,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
    )

    pipeline = RAGPipeline(config)

    # Example questions
    questions = [
        "What is the host of Plasmodium falciparum?",
        "What are the symptoms of malaria?",
    ]

    for i, question in enumerate(questions, 1):
        print("=" * 80)
        print(f"Question {i}/{len(questions)}")
        print("=" * 80)

        result = pipeline.run(question, topic_id=f"Q{i:03d}")

        print("\nRESULT:")
        print(json.dumps(result, indent=2))
        print("\n")


if __name__ == "__main__":
    main()
