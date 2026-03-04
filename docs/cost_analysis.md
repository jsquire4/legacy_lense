# LegacyLens Cost Analysis

## Model Pricing (per 1M tokens)

| Model | Input | Output | Type |
|-------|-------|--------|------|
| GPT-3.5-turbo | $0.50 | $1.50 | Legacy |
| GPT-4o-mini | $0.15 | $0.60 | Standard |
| GPT-4o | $2.50 | $10.00 | Standard |
| GPT-4.1-nano | $0.10 | $0.40 | Standard (default) |
| GPT-4.1-mini | $0.40 | $1.60 | Standard |
| GPT-4.1 | $2.00 | $8.00 | Standard |
| GPT-5-nano | $0.05 | $0.40 | Reasoning |
| GPT-5-mini | $0.25 | $2.00 | Reasoning |
| GPT-5.2 | $1.75 | $14.00 | Reasoning |

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

### Default Model: GPT-4.1-nano

| Component | Tokens | Cost |
|-----------|--------|------|
| Query embedding | ~20 tokens | $0.0000004 |
| Context assembly | ~3,000 tokens (prompt) | — |
| LLM generation (prompt) | ~3,500 tokens | $0.00035 |
| LLM generation (completion) | ~500 tokens | $0.0002 |
| **Total per query** | | **~$0.0006** |

### With LLM Query Expansion

Adds ~$0.00002 for the expansion call (max_tokens=50). Negligible.

### Cost Per Query by Model

| Model | Est. cost/query |
|-------|----------------|
| GPT-3.5-turbo | ~$0.0025 |
| GPT-4o-mini | ~$0.0008 |
| GPT-4o | ~$0.014 |
| GPT-4.1-nano | ~$0.0006 |
| GPT-4.1-mini | ~$0.002 |
| GPT-4.1 | ~$0.011 |
| GPT-5-nano | ~$0.0004 |
| GPT-5-mini | ~$0.002 |
| GPT-5.2 | ~$0.013 |

Assumes ~3,500 prompt tokens + ~500 completion tokens per query.

## Scaling Projections (GPT-4.1-nano default)

| Users | Queries/day (est.) | Monthly Cost | Annual Cost |
|-------|--------------------|--------------|-------------|
| 100 | 50 | $0.90 | $11 |
| 1,000 | 500 | $9 | $108 |
| 10,000 | 5,000 | $90 | $1,080 |
| 100,000 | 50,000 | $900 | $10,800 |

Assumptions: 0.5 queries per user per day, 30 days/month. ~60% of queries trigger expansion.

## Infrastructure Costs

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Qdrant Cloud | 1GB free (sufficient for LAPACK) | $25/mo for 4GB |
| Railway | $5/mo (Hobby) | $20/mo (Pro) |

## Cost Optimization (Already Implemented)

1. **Response caching**: LRU cache (64 entries, 5-min TTL) avoids redundant LLM calls
2. **Cache warming**: 21 pre-cached responses (7 queries x 3 cheap models) on startup
3. **Expansion gating**: Skip LLM expansion for queries that already contain routine names
4. **Model selection**: Users choose cost/quality tradeoff per query
5. **Unified token budgets**: 3000 context / 2048 response — prevents expensive over-generation

## Break-even Analysis

At 100k users with GPT-4.1-nano ($900/mo API + $30/mo infra = $930/mo):
- $0.0093/user/month
- Trivially sustainable at any pricing tier
