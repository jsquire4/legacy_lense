"""Query → embed → vector search pipeline with hybrid name matching."""

import logging
import re

from app.services.embeddings import embed_texts, get_openai_client
from app.services.vector_store import search, search_by_name
from app.config import get_settings

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


def _expand_query(query: str) -> list[str]:
    """Use the LLM to identify relevant LAPACK routine names for a conceptual query."""
    try:
        client = get_openai_client()
        settings = get_settings()
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": _EXPAND_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=50,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        # Only keep tokens that look like LAPACK/BLAS names (start with S/D/C/Z/I prefix)
        all_tokens = re.findall(r'[A-Z][A-Z0-9]{3,}', text.upper())
        names = [n for n in all_tokens if n[0] in "SDCZI" and len(n) >= 4]
        logger.info("Query expansion for '%.60s': %s", query, names)
        return names
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return []


def retrieve(query: str, top_k: int = 8) -> dict:
    """Embed a query and search — with name-boosted hybrid retrieval.

    Returns a dict with keys: chunks, expanded_names, retrieval_strategy.
    """
    embeddings = embed_texts([query])
    if not embeddings:
        logger.error("Failed to embed query")
        return {"chunks": [], "expanded_names": [], "retrieval_strategy": "failed"}

    query_embedding = embeddings[0]
    routine_name = _extract_routine_name(query)

    # Collect name-matched results from explicit name or LLM expansion
    name_results = []
    seen_ids = set()
    expanded_names: list[str] = []
    strategy = "vector"

    if routine_name:
        strategy = "name_match"
        hits = search_by_name(query_embedding, routine_name, top_k=3)
        for h in hits:
            if h["id"] not in seen_ids:
                h["_match_type"] = "name"
                name_results.append(h)
                seen_ids.add(h["id"])
    else:
        # Conceptual query — expand with LLM
        expanded_names = _expand_query(query)
        if expanded_names:
            strategy = "query_expansion"
        for name in expanded_names[:10]:
            hits = search_by_name(query_embedding, name, top_k=1)
            for h in hits:
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
        new_calls = called_names - found_names
        for name in sorted(new_calls)[:5]:
            hits = search_by_name(query_embedding, name, top_k=1)
            for h in hits:
                if h["id"] not in seen_ids:
                    h["_match_type"] = "call_graph"
                    call_graph_results.append(h)
                    seen_ids.add(h["id"])

    # Always do vector search too
    vector_results = search(query_embedding, top_k=top_k)

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
