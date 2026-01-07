"""
Fast RAG Pipeline using vLLM for 1-second inference

This module uses vLLM for 10-100x faster inference compared to HuggingFace Transformers.
"""

from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from vllm import LLM, SamplingParams

from .retrieval import SIBILSRetriever, Document
from .generation import LLMGenerator


@dataclass
class RAGConfig:
    """Configuration for fast RAG pipeline."""
    # Retrieval
    retrieval_n: int = 100
    retrieval_collection: str = "pmc"

    # Reranking
    rerank: bool = False
    rerank_n: int = 20

    # Generation
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True
    gpu_memory_utilization: float = 0.8
    max_new_tokens: int = 512
    temperature: float = 0.7

    # General
    device: str = "cuda"


class FastRAGPipeline:
    """
    Ultra-fast RAG pipeline using vLLM.

    Expected performance:
    - Retrieval: ~2-5 seconds
    - Generation (vLLM): ~0.5-1 second
    - Total: ~3-6 seconds per question
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize fast pipeline."""
        self.config = config or RAGConfig()

        print("Initializing Fast BioMoQA RAG Pipeline...")
        print(f"Config: {self.config}\n")

        # Initialize retriever
        print("Loading retriever...")
        self.retriever = SIBILSRetriever(
            collection=self.config.retrieval_collection,
            default_n=self.config.retrieval_n,
        )

        # Initialize vLLM generator (much faster than HuggingFace)
        if self.config.use_vllm:
            print(f"Loading vLLM model: {self.config.model_name}")
            print("This will be MUCH faster than standard transformers!")

            self.llm = LLM(
                model=self.config.model_name,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=8192,
            )

            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens,
                top_p=0.9,
            )

            print("✓ vLLM loaded successfully!\n")
        else:
            # Fallback to standard generator
            print("Loading standard generator (slower)...")
            self.generator = LLMGenerator(
                model_name=self.config.model_name,
                device=self.config.device,
                load_in_4bit=True,
                max_new_tokens=self.config.max_new_tokens,
            )

        print("Pipeline ready!\n")

    def create_rag_prompt(self, question: str, documents: List[Document]) -> str:
        """Create RAG prompt."""
        context_str = ""
        for i, doc in enumerate(documents):
            title = doc.title
            text = doc.get_text(max_length=1000)
            context_str += f"[{i}] {title}\n{text}\n\n"

        prompt = f"""System: Answer the question using the provided context documents. Cite sources using [0], [1], etc.

QUESTION: {question}

CONTEXTS:
{context_str}

ANSWER:"""

        return prompt

    def parse_citations(self, answer_text: str, documents: List[Document]) -> List[Dict]:
        """Parse answer into sentences with explicit citations."""
        import re

        sentences = []
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer_text)

        for sent in raw_sentences:
            if not sent.strip():
                continue

            citation_pattern = r'\[(\d+)\]'
            matches = re.findall(citation_pattern, sent)
            citation_ids = [int(m) for m in matches]

            # Build explicit citation details
            citation_details = []
            for cid in citation_ids:
                if cid < len(documents):
                    doc = documents[cid]
                    citation_details.append({
                        "document_id": cid,
                        "document_title": doc.title,
                        "pmcid": doc.doc_id,
                    })

            clean_text = re.sub(citation_pattern, '', sent).strip()

            if clean_text:
                sentences.append({
                    "text": clean_text,
                    "citation_ids": citation_ids,
                    "citations": citation_details,
                })

        return sentences

    def generate_vllm(self, question: str, documents: List[Document]) -> Dict:
        """Generate answer using vLLM (fast)."""
        prompt = self.create_rag_prompt(question, documents)

        # vLLM inference (much faster!)
        outputs = self.llm.generate([prompt], self.sampling_params)
        answer_text = outputs[0].outputs[0].text.strip()

        return answer_text

    def run(
        self,
        question: str,
        retrieval_n: Optional[int] = None,
        rerank_n: Optional[int] = None,
        topic_id: Optional[str] = None,
        return_documents: bool = False,
    ) -> Dict:
        """Run fast RAG pipeline."""
        start_time = time.time()

        retrieval_n = retrieval_n or self.config.retrieval_n
        rerank_n = rerank_n or self.config.rerank_n

        # Stage 1: Retrieval
        documents = self.retriever.retrieve(question, n=retrieval_n)

        # Stage 2: Reranking (simple top-k for now)
        documents = documents[:rerank_n]

        # Stage 3: Generation with vLLM
        if self.config.use_vllm:
            answer_text = self.generate_vllm(question, documents)
        else:
            result = self.generator.generate(
                question,
                [{"title": d.title, "text": d.get_text(1000)} for d in documents]
            )
            answer_text = result["answer_text"]

        # Parse citations with explicit details
        answer_sentences = self.parse_citations(answer_text, documents)

        # Build references list with full document info
        references = []
        for i, doc in enumerate(documents):
            references.append(f"[{i}] {doc.doc_id}: {doc.title}")

        # Build output
        output = {
            "topic_id": topic_id or "unknown",
            "question": question,
            "references": references,
            "response_length": len(answer_text),
            "answer": answer_sentences,
            "raw_answer": answer_text,
            "pipeline_time": round(time.time() - start_time, 2),
            "num_retrieved": len(documents),
        }

        if return_documents:
            output["documents"] = [
                {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "score": doc.score,
                    "abstract": doc.abstract[:200],
                }
                for doc in documents
            ]

        return output
