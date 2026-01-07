"""
LLM-based Answer Generator with Citations

This module implements the augmented generation stage (AG) of the Ragnarok RAG pipeline.
It uses open-source LLMs to generate answers with sentence-level citations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import json
import re


class LLMGenerator:
    """
    Answer generator using open-source LLMs with RAG.

    Supports:
    - Llama 3.1 (8B, 70B)
    - Qwen 2.5 (7B, 14B)
    - Mistral 7B Instruct
    - Other instruction-tuned models
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
    ):
        """
        Initialize LLM generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda' or 'cpu')
            load_in_8bit: Load model in 8-bit (reduces memory)
            load_in_4bit: Load model in 4-bit (further reduces memory)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_name}...")

        # Quantization config for memory efficiency
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        print(f"Model loaded successfully on {device}")

    def create_rag_prompt(
        self,
        question: str,
        documents: List[Dict],
        prompt_template: str = "ragnarok",
    ) -> str:
        """
        Create RAG prompt from question and retrieved documents.

        Args:
            question: User question
            documents: Retrieved documents (list of dicts with 'title' and 'text')
            prompt_template: Prompt style ('ragnarok', 'chatqa', 'simple')

        Returns:
            Formatted prompt string
        """
        # Format documents with citations
        context_str = ""
        for i, doc in enumerate(documents):
            title = doc.get("title", "Untitled")
            text = doc.get("text", doc.get("abstract", ""))
            context_str += f"[{i}] {title}\n{text}\n\n"

        if prompt_template == "ragnarok" or prompt_template == "chatqa":
            # Ragnarok/ChatQA style (from paper)
            prompt = f"""System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.

INSTRUCTION: Please give a complete answer to the question. Cite each context document that supports your answer within brackets [] using the format [0], [1], etc.

QUESTION: {question}

CONTEXTS:
{context_str}

INSTRUCTION: Please give a complete answer to the question. Cite each context document that supports your answer within brackets [] using the format [0], [1], etc.

ANSWER:"""

        else:  # simple
            prompt = f"""Answer the following question using the provided context documents. Cite sources using [0], [1], etc.

Question: {question}

Context:
{context_str}

Answer:"""

        return prompt

    def generate(
        self,
        question: str,
        documents: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict:
        """
        Generate answer with citations.

        Args:
            question: User question
            documents: Retrieved documents
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dict with 'answer_text' and 'raw_output'
        """
        # Create prompt
        prompt = self.create_rag_prompt(question, documents)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return {
            "answer_text": generated_text.strip(),
            "raw_output": generated_text,
        }

    def parse_citations(self, answer_text: str) -> List[Dict]:
        """
        Parse answer into sentences with citations (Ragnarok format).

        Args:
            answer_text: Generated answer with citations like [0], [1]

        Returns:
            List of dicts with 'text' and 'citations' (Ragnarok format)
        """
        sentences = []

        # Split into sentences (simple split on periods)
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer_text)

        for sent in raw_sentences:
            if not sent.strip():
                continue

            # Extract citations [0], [1], etc.
            citations = []
            citation_pattern = r'\[(\d+)\]'
            matches = re.findall(citation_pattern, sent)
            citations = [int(m) for m in matches]

            # Remove citations from text
            clean_text = re.sub(citation_pattern, '', sent).strip()

            if clean_text:
                sentences.append({
                    "text": clean_text,
                    "citations": citations,
                })

        return sentences

    def generate_ragnarok_output(
        self,
        question: str,
        documents: List[Dict],
        topic_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate full Ragnarok-format output.

        Args:
            question: User question
            documents: Retrieved documents
            topic_id: Optional topic identifier

        Returns:
            Dict in Ragnarok JSON format
        """
        # Generate answer
        result = self.generate(question, documents)
        answer_text = result["answer_text"]

        # Parse into sentences with citations
        answer_sentences = self.parse_citations(answer_text)

        # Format references
        references = [f"doc{i}" for i in range(len(documents))]

        # Build output
        output = {
            "topic_id": topic_id or "unknown",
            "question": question,
            "references": references,
            "response_length": len(answer_text),
            "answer": answer_sentences,
            "raw_answer": answer_text,  # For debugging
        }

        return output


def main():
    """Example usage of LLM generator."""
    from src.retrieval import SIBILSRetriever

    # Initialize retriever and generator
    print("Initializing components...")
    retriever = SIBILSRetriever()

    # Note: This will download ~8GB model on first run
    generator = LLMGenerator(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
    )

    # Example question
    question = "What is the host of Plasmodium falciparum?"
    print(f"\nQuestion: {question}\n")

    # Retrieve documents
    print("Retrieving documents...")
    docs = retriever.retrieve(question, n=20)

    # Format for generator
    doc_dicts = [
        {
            "title": doc.title,
            "text": doc.abstract[:500],  # Truncate for demo
        }
        for doc in docs[:5]  # Use top 5 for demo
    ]

    # Generate answer
    print("Generating answer...")
    output = generator.generate_ragnarok_output(
        question=question,
        documents=doc_dicts,
        topic_id="Q001",
    )

    # Display results
    print("\n" + "="*80)
    print("RAGNAROK OUTPUT")
    print("="*80)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
