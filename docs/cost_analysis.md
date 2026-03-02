# LegacyLens Cost Analysis

## Embedding Costs (One-time Ingestion)

| Metric | Value |
|--------|-------|
| Total chunks | ~2,334 |
| Avg tokens per chunk | ~2,000 |
| Total tokens | ~4.7M |
| Model | text-embedding-3-small |
| Cost per 1M tokens | $0.020 |
| **Total embedding cost** | **~$0.09** |

## Query Costs (Per Query)

### Standard Query (name match or vector only)

| Component | Tokens | Cost |
|-----------|--------|------|
| Query embedding | ~20 tokens | $0.0000004 |
| Context assembly | ~6,000 tokens (prompt) | — |
| LLM generation (prompt) | ~6,500 tokens | $0.00098 |
| LLM generation (completion) | ~500 tokens | $0.0006 |
| **Total per query** | | **~$0.0016** |

### Expanded Query (LLM query expansion triggered)

| Component | Tokens | Cost |
|-----------|--------|------|
| Query embedding | ~20 tokens | $0.0000004 |
| Expansion LLM call (prompt) | ~150 tokens | $0.0000225 |
| Expansion LLM call (completion) | ~30 tokens | $0.000018 |
| Context assembly | ~6,000 tokens (prompt) | — |
| LLM generation (prompt) | ~6,500 tokens | $0.00098 |
| LLM generation (completion) | ~500 tokens | $0.0006 |
| **Total per expanded query** | | **~$0.0016** |

Model: gpt-4o-mini ($0.150/1M input, $0.600/1M output)

Note: Query expansion adds negligible cost (~$0.00004) because the expansion prompt is very small (max_tokens=50).

## Scaling Projections

| Users | Queries/day (est.) | Monthly Cost | Annual Cost |
|-------|--------------------|--------------|-------------|
| 100 | 50 | $2.40 | $29 |
| 1,000 | 500 | $24 | $288 |
| 10,000 | 5,000 | $240 | $2,880 |
| 100,000 | 50,000 | $2,400 | $28,800 |

Assumptions: 0.5 queries per user per day, 30 days/month. ~60% of queries trigger expansion.

## Infrastructure Costs

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Qdrant Cloud | 1GB free (sufficient for LAPACK) | $25/mo for 4GB |
| Railway | $5/mo (Hobby) | $20/mo (Pro) |

## Cost Optimization Opportunities

1. **Caching**: Cache frequent queries to avoid redundant LLM calls (estimated 30-50% hit rate)
2. **Smaller context**: Reduce top_k from 8 to 5 for simple queries
3. **Batch queries**: Group similar queries for embedding efficiency
4. **Model selection**: Use gpt-4o-mini (current) vs gpt-4o for cost-sensitive deployments
5. **Expansion gating**: Skip LLM expansion for queries that already contain routine names (already implemented)

## Break-even Analysis

At 100k users ($2,400/mo API + $45/mo infra = $2,445/mo):
- $0.024/user/month
- Sustainable with a $1/mo subscription or ad-supported model
