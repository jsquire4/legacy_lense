"""Query → embed → vector search pipeline with hybrid name matching."""

import asyncio
import logging
import re

from app.services.embeddings import embed_query, get_async_openai_client
from app.services.vector_store import async_search, async_search_by_name
from app.config import get_settings
from app.models_data import is_reasoning_model

logger = logging.getLogger(__name__)

# Pattern to detect LAPACK/BLAS routine names in queries (case-sensitive, min 4 chars after prefix)
_ROUTINE_NAME_RE = re.compile(r'\b([SDCZI][A-Z][A-Z0-9]{3,})\b')

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


def _extract_routine_name(query: str) -> str | None:
    """Extract a probable LAPACK/BLAS routine name from the query."""
    match = _ROUTINE_NAME_RE.search(query)
    if match:
        return match.group(1).upper()
    return None


async def _expand_query(query: str, model: str | None = None) -> list[str]:
    """Use the LLM to identify relevant LAPACK routine names for a conceptual query."""
    try:
        client = get_async_openai_client()
        settings = get_settings()
        resolved_model = model or settings.CHAT_MODEL
        kwargs = dict(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _EXPAND_PROMPT},
                {"role": "user", "content": query},
            ],
            max_completion_tokens=50,
        )
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
        logger.warning("Query expansion failed: %s", e)
        return []


async def _fan_out_name_search(
    query_embedding: list[float], names: list[str], top_k_each: int,
) -> list[dict]:
    """Run async_search_by_name for multiple names concurrently."""
    if not names:
        return []
    tasks = [
        async_search_by_name(query_embedding, name, top_k=top_k_each)
        for name in names
    ]
    results_list = await asyncio.gather(*tasks)
    results = []
    for hits in results_list:
        results.extend(hits)
    return results


async def retrieve(query: str, top_k: int = 5, model: str | None = None) -> dict:
    """Embed a query and search — with name-boosted hybrid retrieval.

    Returns a dict with keys: chunks, expanded_names, retrieval_strategy.
    """
    query_embedding = await embed_query(query)
    if not query_embedding:
        logger.error("Failed to embed query")
        return {"chunks": [], "expanded_names": [], "retrieval_strategy": "failed"}

    routine_name = _extract_routine_name(query)

    # Collect name-matched results from explicit name or LLM expansion
    name_results = []
    seen_ids = set()
    expanded_names: list[str] = []
    strategy = "vector"

    if routine_name:
        strategy = "name_match"
        hits = await async_search_by_name(query_embedding, routine_name, top_k=3)
        for h in hits:
            if h["id"] not in seen_ids:
                h["_match_type"] = "name"
                name_results.append(h)
                seen_ids.add(h["id"])
    else:
        # Conceptual query — LLM expansion to find relevant routine names
        expanded_names = await _expand_query(query, model=model)
        if expanded_names:
            strategy = "expansion"
            expansion_hits = await _fan_out_name_search(
                query_embedding, expanded_names, top_k_each=2,
            )
            for h in expansion_hits:
                if h["id"] not in seen_ids:
                    h["_match_type"] = "expansion"
                    name_results.append(h)
                    seen_ids.add(h["id"])

    # Follow call graph one hop: find routines called by name-matched results
    call_graph_results = []
    if name_results:
        called_names = set()
        for r in name_results:
            for called in r["metadata"].get("called_routines", []):
                called_names.add(called)
        # Only follow calls to routines we haven't already found
        found_names = {r["metadata"].get("unit_name") for r in name_results}
        new_calls = sorted(called_names - found_names)[:5]

        # Fan out call-graph searches concurrently
        hits_list = await _fan_out_name_search(query_embedding, new_calls, top_k_each=1)
        for h in hits_list:
            if h["id"] not in seen_ids:
                h["_match_type"] = "call_graph"
                call_graph_results.append(h)
                seen_ids.add(h["id"])

    # Vector search (conceptual queries use vector only; name-match adds vector results)
    vector_results = await async_search(query_embedding, top_k=top_k)

    # Merge: name-matched first, then call-graph, then vector (deduplicated)
    merged = list(name_results) + call_graph_results
    for r in vector_results:
        if r["id"] not in seen_ids and len(merged) < top_k:
            r.setdefault("_match_type", "vector")
            merged.append(r)
            seen_ids.add(r["id"])

    logger.info(
        "Retrieved %d results (%d name-matched, %d call-graph) for query: %.80s",
        len(merged), len(name_results), len(call_graph_results), query,
    )
    return {
        "chunks": merged,
        "expanded_names": expanded_names,
        "retrieval_strategy": strategy,
    }
