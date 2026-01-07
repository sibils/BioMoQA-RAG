"""
Query Expansion for Better Retrieval

Expands queries with synonyms, acronym expansions, and alternative phrasings
to improve retrieval recall.
"""

from typing import List, Set
from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    """Expanded query with alternatives."""
    original: str
    expansions: List[str]
    all_queries: List[str]  # original + expansions


class LLMQueryExpander:
    """
    Use LLM to expand queries with biomedical synonyms and acronym expansions.

    This helps retrieve documents that use different terminology than the query.
    """

    def __init__(self, llm=None, sampling_params=None):
        """
        Initialize query expander.

        Args:
            llm: vLLM LLM instance (optional, will use provided one)
            sampling_params: SamplingParams for generation
        """
        self.llm = llm
        self.sampling_params = sampling_params

    def expand(self, question: str, n_variants: int = 2) -> ExpandedQuery:
        """
        Expand query with alternative phrasings.

        Args:
            question: Original question
            n_variants: Number of alternative phrasings to generate

        Returns:
            ExpandedQuery with original + alternatives
        """
        if self.llm is None:
            # No LLM available, return original only
            return ExpandedQuery(
                original=question,
                expansions=[],
                all_queries=[question]
            )

        # Prompt for query expansion
        prompt = f"""Given this biomedical question: "{question}"

Generate {n_variants} alternative phrasings that:
1. Expand any acronyms (e.g., "AG1-IA" -> "anastomosis group 1 IA")
2. Use medical synonyms (e.g., "host" -> "reservoir organism", "causes" -> "etiology")
3. Rephrase using technical terminology

Format: one alternative per line, no numbering, no explanations.

Alternatives:"""

        # Generate with LLM
        from vllm import SamplingParams
        expansion_params = SamplingParams(
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
        )

        outputs = self.llm.generate([prompt], expansion_params)
        response = outputs[0].outputs[0].text.strip()

        # Parse alternatives (one per line)
        expansions = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering if present
            if line and len(line) > 5:
                # Remove common prefixes like "1.", "2)", "-", "*"
                cleaned = line.lstrip('0123456789.-*) ')
                if cleaned and cleaned != question:
                    expansions.append(cleaned)

        # Limit to n_variants
        expansions = expansions[:n_variants]

        return ExpandedQuery(
            original=question,
            expansions=expansions,
            all_queries=[question] + expansions
        )


class AcronymExpander:
    """
    Simple rule-based acronym expansion for common biomedical terms.

    This is faster than LLM-based expansion and works well for known acronyms.
    """

    def __init__(self):
        """Initialize with common biomedical acronyms."""
        self.acronym_map = {
            # Organisms
            'AG1-IA': 'anastomosis group 1 IA',
            'P. falciparum': 'Plasmodium falciparum',
            'E. coli': 'Escherichia coli',
            'S. aureus': 'Staphylococcus aureus',

            # Diseases
            'BLSB': 'banded leaf and sheath blight',
            'PMC': 'PubMed Central',

            # General
            'DNA': 'deoxyribonucleic acid',
            'RNA': 'ribonucleic acid',
            'PCR': 'polymerase chain reaction',
        }

    def expand(self, question: str) -> ExpandedQuery:
        """
        Expand known acronyms in the question.

        Args:
            question: Original question

        Returns:
            ExpandedQuery with acronym expansions
        """
        expansions = []

        # Check if any acronyms are in the question
        for acronym, expansion in self.acronym_map.items():
            if acronym in question:
                # Create variant with expansion
                expanded = question.replace(acronym, expansion)
                if expanded != question:
                    expansions.append(expanded)

        return ExpandedQuery(
            original=question,
            expansions=expansions,
            all_queries=[question] + expansions
        )


class HybridQueryExpander:
    """
    Combine rule-based and LLM-based expansion.

    Uses fast rule-based expansion for known terms, LLM for others.
    """

    def __init__(self, llm=None, sampling_params=None):
        """Initialize hybrid expander."""
        self.acronym_expander = AcronymExpander()
        self.llm_expander = LLMQueryExpander(llm, sampling_params)

    def expand(self, question: str, n_llm_variants: int = 1) -> ExpandedQuery:
        """
        Expand with both rule-based and LLM methods.

        Args:
            question: Original question
            n_llm_variants: Number of LLM-generated variants

        Returns:
            ExpandedQuery with all expansions
        """
        # First: rule-based expansion
        acronym_result = self.acronym_expander.expand(question)

        # Then: LLM expansion (if available)
        llm_result = self.llm_expander.expand(question, n_llm_variants)

        # Combine (deduplicate)
        all_queries = [question]
        seen = {question.lower()}

        for query in acronym_result.expansions + llm_result.expansions:
            if query.lower() not in seen:
                all_queries.append(query)
                seen.add(query.lower())

        return ExpandedQuery(
            original=question,
            expansions=all_queries[1:],  # Everything except original
            all_queries=all_queries
        )
