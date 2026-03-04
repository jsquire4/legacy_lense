"""Query → embed → vector search pipeline with hybrid name matching."""

import asyncio
import logging
import re

from app.services.embeddings import embed_query, get_async_openai_client
from app.services.vector_store import async_search, async_search_by_name, async_search_by_caller
from app.config import get_settings
from app.models_data import is_reasoning_model, uses_legacy_max_tokens

logger = logging.getLogger(__name__)

# Pattern to detect LAPACK/BLAS routine names in queries (case-sensitive, min 4 chars after prefix)
_ROUTINE_NAME_RE = re.compile(r'\b([SDCZI][A-Z][A-Z0-9]{2,})\b')

_EXPAND_PROMPT = (
    "You are a LAPACK/BLAS expert. Given a user question, list the 5-8 most relevant "
    "double-precision LAPACK or BLAS routine names. List the PRIMARY driver first, then "
    "its closest variants. Prioritize diversity across algorithm variants over listing all "
    "matrix-type specializations. Examples:\n"
    "- SVD: DGESVD DGESDD DGESVDX\n"
    "- LU: DGETRF DGETRF2 DGESV\n"
    "- Eigenvalues: DGEEV DSYEV DSYEVD DSYEVR\n"
    "- Cholesky: DPOTRF DPOTRF2 DPOTF2 DPOSV\n"
    "Return ONLY routine names separated by spaces, nothing else."
)


_ROUTINE_NAME_RE_CI = re.compile(r'\b([sdcziSDCZI][a-zA-Z][a-zA-Z0-9]{2,})\b')

# Common English words that match the LAPACK prefix pattern (S/D/C/Z/I + letter + 2+ alphanum)
_ENGLISH_STOPWORDS = frozenset({
    "DOES", "SOME", "SAME", "SIZE", "SUCH", "SHOW", "SHOULD", "SINCE",
    "CALL", "CODE", "CORE", "DATA", "DUMP", "EACH",
    "INTO", "CASE", "SAID", "STEP", "SORT", "SCAN", "COPY",
    "SKIP", "COME", "DONE", "DARE", "IDEA", "ITEM",
})


def _extract_routine_name(query: str) -> str | None:
    """Extract a probable LAPACK/BLAS routine name from the query."""
    # Try strict uppercase first (avoids false positives like "Does", "Simple")
    match = _ROUTINE_NAME_RE.search(query)
    if match:
        return match.group(1).upper()
    # Fallback: case-insensitive match for lowercase queries like "explain dgesv"
    # LAPACK/BLAS names are 4-8 chars; reject longer words (likely English)
    for match in _ROUTINE_NAME_RE_CI.finditer(query):
        candidate = match.group(1).upper()
        if 4 <= len(candidate) <= 8 and candidate not in _ENGLISH_STOPWORDS:
            return candidate
    return None


async def _expand_query(query: str, model: str | None = None) -> list[str]:
    """Use the LLM to identify relevant LAPACK routine names for a conceptual query."""
    try:
        client = get_async_openai_client()
        settings = get_settings()
        resolved_model = model or settings.CHAT_MODEL
        token_key = "max_tokens" if uses_legacy_max_tokens(resolved_model) else "max_completion_tokens"
        kwargs = dict(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _EXPAND_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        kwargs[token_key] = 50
        if is_reasoning_model(resolved_model):
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["temperature"] = 0
        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        # Only keep tokens that look like LAPACK/BLAS names (start with S/D/C/Z/I prefix)
        all_tokens = re.findall(r'[A-Z][A-Z0-9]{3,}', text.upper())
        names = [n for n in all_tokens if n[0] in "SDCZI" and len(n) >= 4]
        logger.info("Query expansion for '%.60s': %s", query, names)
        return names
    except Exception as e:
        logger.warning("Query expansion failed for '%.60s': %s", query, e)
        return []


async def _fan_out_name_search(
    query_embedding: list[float], names: list[str], top_k_each: int,
    collection_name: str | None = None,
) -> list[dict]:
    """Run async_search_by_name for multiple names concurrently."""
    if not names:
        return []
    tasks = [
        async_search_by_name(query_embedding, name, top_k=top_k_each,
                             collection_name=collection_name)
        for name in names
    ]
    results_list = await asyncio.gather(*tasks)
    results = []
    for hits in results_list:
        results.extend(hits)
    return results


async def _name_match_search(
    query_embedding: list[float], routine_name: str, seen_ids: set,
    collection_name: str | None = None,
) -> list[dict]:
    """Direct name lookup for an explicitly mentioned routine."""
    results = []
    hits = await async_search_by_name(query_embedding, routine_name, top_k=3,
                                      collection_name=collection_name)
    for h in hits:
        if h["id"] not in seen_ids:
            h["_match_type"] = "name"
            results.append(h)
            seen_ids.add(h["id"])
    return results


async def _expansion_search(
    query: str, query_embedding: list[float], seen_ids: set,
    model: str | None = None, collection_name: str | None = None,
) -> tuple[list[dict], list[str]]:
    """LLM expansion to find relevant routine names for conceptual queries."""
    expanded_names = await _expand_query(query, model=model)
    results = []
    if expanded_names:
        expansion_hits = await _fan_out_name_search(
            query_embedding, expanded_names, top_k_each=2,
            collection_name=collection_name,
        )
        for h in expansion_hits:
            if h["id"] not in seen_ids:
                h["_match_type"] = "expansion"
                results.append(h)
                seen_ids.add(h["id"])
    return results, expanded_names


async def _call_graph_search(
    query_embedding: list[float], name_results: list[dict], seen_ids: set,
    collection_name: str | None = None,
) -> list[dict]:
    """Follow call graph one hop: find routines called by name-matched results."""
    results = []
    if not name_results:
        return results
    called_names = set()
    for r in name_results:
        for called in r["metadata"].get("called_routines", []):
            called_names.add(called)
    found_names = {r["metadata"].get("unit_name") for r in name_results}
    new_calls = sorted(called_names - found_names)[:5]

    hits_list = await _fan_out_name_search(query_embedding, new_calls, top_k_each=1,
                                           collection_name=collection_name)
    for h in hits_list:
        if h["id"] not in seen_ids:
            h["_match_type"] = "call_graph"
            results.append(h)
            seen_ids.add(h["id"])
    return results


async def _caller_search(
    query_embedding: list[float], routine_name: str, seen_ids: set,
    collection_name: str | None = None,
) -> list[dict]:
    """Impact analysis: find routines that CALL the target routine."""
    results = []
    caller_hits = await async_search_by_caller(query_embedding, routine_name, top_k=5,
                                               collection_name=collection_name)
    for h in caller_hits:
        if h["id"] not in seen_ids:
            h["_match_type"] = "called_by"
            results.append(h)
            seen_ids.add(h["id"])
    return results


async def retrieve(query: str, top_k: int = 5, model: str | None = None, capability: str | None = None,
                   collection_name: str | None = None, embedding_model: str | None = None,
                   expanded_names: list[str] | None = None) -> dict:
    """Embed a query and search — with name-boosted hybrid retrieval.

    Returns a dict with keys: chunks, expanded_names, retrieval_strategy.
    """
    query_embedding = await embed_query(query, model=embedding_model)
    if not query_embedding:
        logger.error("Failed to embed query")
        return {"chunks": [], "expanded_names": [], "retrieval_strategy": "failed"}

    routine_name = _extract_routine_name(query)
    seen_ids: set = set()
    resolved_expanded: list[str] = []
    strategy = "vector"

    if routine_name:
        strategy = "name_match"
        name_results = await _name_match_search(query_embedding, routine_name, seen_ids,
                                                collection_name=collection_name)
    else:
        if expanded_names:
            # Pre-fetched — skip the LLM expansion call entirely
            logger.info("Using prefetched expansion: %s", expanded_names)
            expansion_hits = await _fan_out_name_search(
                query_embedding, expanded_names, top_k_each=2,
                collection_name=collection_name,
            )
            name_results = []
            for h in expansion_hits:
                if h["id"] not in seen_ids:
                    h["_match_type"] = "expansion"
                    name_results.append(h)
                    seen_ids.add(h["id"])
            resolved_expanded = expanded_names
            strategy = "expansion"
        else:
            name_results, resolved_expanded = await _expansion_search(
                query, query_embedding, seen_ids, model=model, collection_name=collection_name,
            )
            if resolved_expanded:
                strategy = "expansion"

    call_graph_results = await _call_graph_search(query_embedding, name_results, seen_ids,
                                                  collection_name=collection_name)

    caller_results = []
    if capability == "impact_analysis" and routine_name:
        caller_results = await _caller_search(query_embedding, routine_name, seen_ids,
                                              collection_name=collection_name)

    vector_results = await async_search(query_embedding, top_k=top_k,
                                        collection_name=collection_name)

    merged = list(name_results) + call_graph_results + caller_results
    for r in vector_results:
        if r["id"] not in seen_ids and len(merged) < top_k:
            r.setdefault("_match_type", "vector")
            merged.append(r)
            seen_ids.add(r["id"])

    logger.info(
        "Retrieved %d results (%d name-matched, %d call-graph, %d callers) for query: %.80s",
        len(merged), len(name_results), len(call_graph_results), len(caller_results), query,
    )
    return {
        "chunks": merged,
        "expanded_names": resolved_expanded,
        "retrieval_strategy": strategy,
    }
