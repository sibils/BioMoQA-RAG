"""
SIBILS Query Parser Integration

This module integrates with the SIBILS query parser API to transform
natural language questions into optimized Elasticsearch queries with
concept annotations and expansions.
"""

import requests
from typing import Dict, Optional, List
from dataclasses import dataclass
import time


@dataclass
class ParsedQuery:
    """Represents a parsed query with annotations and ES query."""
    original_query: str
    normalized_query: str
    es_query: Optional[Dict] = None
    text_parts: Optional[List[str]] = None
    json_query: Optional[Dict] = None
    success: bool = True
    error: Optional[str] = None


class SIBILSQueryParser:
    """
    Query parser using SIBILS query parser API.

    The query parser:
    - Annotates biomedical concepts (MeSH, NCIT, AGROVOC)
    - Expands queries with ontology terms
    - Generates optimized Elasticsearch queries
    - Normalizes query syntax

    Example:
        "What causes malaria?" ->
        "causes (malaria OR [ncit:C34797] OR [mesh:D008288] OR [agrovoc:c_34312])"
    """

    def __init__(
        self,
        api_url: str = "https://biodiversitypmc.dev.sibils.org/api/query/parse",
        collection: str = "pmc",
        timeout: int = 10,
        cache_enabled: bool = True,
    ):
        """
        Initialize SIBILS query parser.

        Args:
            api_url: SIBILS query parser API endpoint
            collection: Target collection for ES query generation ("pmc", "medline", etc.)
            timeout: Request timeout in seconds
            cache_enabled: Enable in-memory caching of parsed queries
        """
        self.api_url = api_url
        self.collection = collection
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, ParsedQuery] = {}

    def parse(
        self,
        query: str,
        collection: Optional[str] = None,
        normalized: bool = True,
        include_es_query: bool = True,
        include_text_parts: bool = True,
    ) -> ParsedQuery:
        """
        Parse a natural language query into a structured query.

        Args:
            query: Natural language question or query
            collection: Override default collection (e.g., "pmc", "medline")
            normalized: Return normalized query with annotations
            include_es_query: Generate Elasticsearch query
            include_text_parts: Extract text parts from query

        Returns:
            ParsedQuery object with normalized query, ES query, and metadata
        """
        # Check cache
        cache_key = f"{query}:{collection or self.collection}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        collection = collection or self.collection

        # Build request parameters
        params = {
            "query": query,
            "normalized": normalized,
            "text_parts": include_text_parts,
        }

        # Add collection for ES query generation
        if include_es_query:
            params["es"] = collection

        try:
            response = requests.get(
                self.api_url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                error = data.get("error", "Query parsing failed")
                return ParsedQuery(
                    original_query=query,
                    normalized_query=query,
                    success=False,
                    error=error,
                )

            # Extract ES query for the specified collection
            es_queries = data.get("es_queries", {})
            es_query = es_queries.get(collection) if es_queries else None

            # Clean ES query by removing punctuation from query strings
            if es_query:
                es_query = self._clean_es_query(es_query)

            parsed = ParsedQuery(
                original_query=query,
                normalized_query=data.get("normalized_query", query),
                es_query=es_query,
                text_parts=data.get("text_parts"),
                json_query=data.get("json_query"),
                success=True,
            )

            # Cache the result
            if self.cache_enabled:
                self._cache[cache_key] = parsed

            return parsed

        except requests.exceptions.RequestException as e:
            return ParsedQuery(
                original_query=query,
                normalized_query=query,
                success=False,
                error=f"API request failed: {str(e)}",
            )

    def _clean_es_query(self, es_query: Dict) -> Dict:
        """
        Clean Elasticsearch query by removing punctuation-only clauses.

        Args:
            es_query: Elasticsearch query dictionary

        Returns:
            Cleaned query dictionary
        """
        import copy

        cleaned = copy.deepcopy(es_query)
        punctuation_to_remove = set('?!.,;:')

        def is_punctuation_only(text: str) -> bool:
            """Check if text contains only punctuation or whitespace."""
            return all(c in punctuation_to_remove or c.isspace() for c in text)

        def clean_query_value(obj):
            """Recursively clean query strings and remove punctuation-only clauses."""
            if isinstance(obj, dict):
                keys_to_remove = []
                for key, value in obj.items():
                    if key == 'query' and isinstance(value, str):
                        # Check if query is punctuation-only
                        if is_punctuation_only(value):
                            # Mark parent dict for removal
                            return None
                    elif isinstance(value, (dict, list)):
                        result = clean_query_value(value)
                        if result is None:
                            keys_to_remove.append(key)
                        else:
                            obj[key] = result

                # Remove keys marked for deletion
                for key in keys_to_remove:
                    del obj[key]

                # If dict is now empty, mark for removal
                if not obj:
                    return None

                return obj

            elif isinstance(obj, list):
                # Filter out None values and empty dicts (punctuation-only clauses)
                cleaned_list = []
                for item in obj:
                    result = clean_query_value(item)
                    # Skip None and empty dicts
                    if result is not None and (not isinstance(result, dict) or result):
                        cleaned_list.append(result)
                return cleaned_list if cleaned_list else None

            return obj

        result = clean_query_value(cleaned)
        return result if result is not None else cleaned

    def clear_cache(self):
        """Clear the query cache."""
        self._cache.clear()


def main():
    """Example usage of SIBILS query parser."""
    parser = SIBILSQueryParser()

    # Example questions
    questions = [
        "What causes malaria?",
        "How does the immune system respond to viral infections?",
        "What are the symptoms of COVID-19?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)

        parsed = parser.parse(question)

        if parsed.success:
            print(f"Normalized: {parsed.normalized_query}")
            print(f"\nText parts: {', '.join(parsed.text_parts or [])}")

            if parsed.es_query:
                print(f"\nES Query generated: Yes")
                # Show simplified ES query structure
                query_type = list(parsed.es_query.get("query", {}).keys())[0]
                print(f"Query type: {query_type}")
            else:
                print("\nES Query generated: No")
        else:
            print(f"Error: {parsed.error}")

        print()


if __name__ == "__main__":
    main()
