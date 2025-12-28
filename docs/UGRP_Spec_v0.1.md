# User-Governed Recommender Playground (UGRP)
**Spec v0.1 — 2025-12-28**  
A reproducible demo + benchmark for **user-controllable**, **faithful-explainable** recommendations, with **apples-to-apples evaluation across LLMs** (e.g., GPT vs Gemini) on public datasets.

---

## 1) Goals and Non-Goals

### 1.1 Goals (must-have in v0.1)
1. **Controllable ranking**  
   Users express preferences/constraints in natural language; the system converts them into an executable control signal that *actually changes* the ranking in a deterministic way.

2. **Faithful explanations**  
   Explanations must be grounded in auditable evidence produced by the ranking pipeline (score decomposition + constraint checks). No “free invention.”

3. **Benchmarkable across LLMs**  
   A standardized test set (**ControlBench**) and automated metrics produce a leaderboard comparing models on:
   - Intent parsing correctness
   - Control effectiveness
   - Explanation faithfulness

### 1.2 Non-goals (explicitly out of scope for v0.1)
- SOTA recommendation accuracy (base recommender is fixed; we’re not optimizing the recommender itself)
- Online A/B tests or real-user deployments
- Long-horizon conversational strategy learning (keep to single-turn or short-turn with optional state)

---

## 2) High-Level System Overview

### 2.1 Data choice (v0.1)
- Primary dataset: **MovieLens** (recommend ML-1M or ML-20M depending on local compute)

### 2.2 Core design principle
- **LLM ≠ ranker.**  
  LLM is used only for:
  1) translating natural language → structured controls (strict JSON), and  
  2) rendering evidence → human-readable explanations.  
- **Ranking and constraint enforcement are deterministic.**  
  Given the same candidates + control JSON, output is identical. This makes cross-LLM evaluation fair.

### 2.3 Online request flow (single interaction)
```
User history ──> Profile Builder ──> Profile Card (UI top)

User message ──> LLM Intent Parser ──> Control JSON ──> Deterministic Re-ranker ──> Top-K
                                          |
                                          └──> Evidence Builder (score breakdown, constraint checks)
                                                   |
                                                   └──> LLM Explanation Renderer (evidence-bound)
                                                            |
                                                            └──> Audit Logger (JSONL replay)
```

---

## 3) User Experience Spec (UI)

### 3.1 Page layout
1. **Profile Card (top panel)**  
   - “You tend to like / avoid …”
   - “Exploration tendency: conservative / medium / adventurous”
   - Evidence: recent liked items (clickable)

2. **Explain Panel (middle panel)**  
   - Tab A: **Control JSON** (collapsed by default; show key fields)
   - Tab B: **Delta summary** (what changed this turn)
   - Tab C: **Filtering & reranking stats** (candidate counts, constraint hits)

3. **Recommendation List (main panel)**  
   - Top-K item cards, each expandable to show:
     - score breakdown (base / control bonus / penalties / diversity adjustment)
     - matched evidence fields (genres, year, popularity quantile)
   - Button: **Counterfactual compare** (“what if you didn’t apply this control?”)

4. **Chat Box (bottom)**  
   - User input
   - Model output: recommended items + explanation + optional “next exploration” prompts

### 3.2 Two modes (optional but recommended)
- **recommend**: return Top-K
- **explore**: return 2–5 “exploration anchors” (diverse suggestions + rationale)

---

## 4) Control Layer: Control JSON (v0.1)

### 4.1 JSON schema (example)
```json
{
  "version": "0.1",
  "intent": {
    "mode": "recommend",
    "k": 10
  },
  "constraints": {
    "include_genres": ["Comedy", "Drama"],
    "exclude_genres": ["Horror"],
    "year_min": 1990,
    "year_max": 2015
  },
  "preferences": {
    "genre_weights": {"Comedy": 0.3, "Drama": 0.2},
    "novelty": 0.6,
    "diversity": 0.5,
    "popularity_bias": -0.2
  },
  "ui": {
    "show_steps": true
  },
  "meta": {
    "confidence": 0.78,
    "notes": "optional short note"
  }
}
```

### 4.2 Field semantics (must map to deterministic computations)
- `exclude_genres`, `year_min/year_max`: **hard constraints** (filter)
- `include_genres`: can be hard (“must contain any”) or soft (bonus). For v0.1, prefer **hard** to simplify evaluation.
- `genre_weights`: **soft preference** bonus
- `novelty` (0..1): encourages long-tail / unseen but similar items
- `diversity` (0..1): encourages variety in the final list
- `popularity_bias` (-1..1): + favors popular items; - favors long-tail

> v0.1 recommendation: avoid attributes not present in MovieLens (e.g., runtime/director) unless you add external metadata. Otherwise you’ll penalize explanation faithfulness.

---

## 5) Base Recommender (Fixed Baseline)

### 5.1 Candidate generation
- Train a simple **ALS/MF** model (or LightFM/BPR).
- For each user, generate Top-N candidates (suggest N = 200).

### 5.2 Metrics (sanity only)
- Report standard offline metrics (Recall@K, NDCG@K) as a “base is functioning” check.
- Do **not** optimize base as the core contribution.

---

## 6) Deterministic Re-ranker (Control Execution)

### 6.1 Step 1: filtering (hard constraints)
Given base candidates:
- Drop items matching `exclude_genres`.
- Drop items outside `[year_min, year_max]`.
- If `include_genres` is hard: keep only items matching at least one included genre.

### 6.2 Step 2: scoring (soft preferences)
Normalize `base_score` to [0, 1]. Compute:
- `genre_bonus = sum(genre_weights[g] for g in item.genres)`
- `pop_term = popularity_quantile(item)` (0..1)
- `novel_term`: long-tail preference signal (simple option: `1 - pop_term`)
- combined:
```
score = base_score
      + α * genre_bonus
      + β * popularity_bias * pop_term
      + γ * novelty * novel_term
```

### 6.3 Step 3: diversity (MMR / greedy)
Use a simple MMR-style greedy selection:
```
select = []
while len(select) < K:
  pick item i maximizing:
    λ * score(i) - (1-λ) * max_{j in select} sim(i, j)
```
- `sim(i, j)` for v0.1 can be **genre Jaccard** or cosine between MF embeddings.
- Map `diversity` knob to λ (or to the penalty weight).

---

## 7) Explanation System (Evidence + LLM Rendering)

### 7.1 Structured evidence (auditable; required)
For every output item, produce a machine-readable explanation:
```json
{
  "item_id": 123,
  "final_score": 0.83,
  "components": {
    "base": 0.72,
    "genre_bonus": 0.08,
    "popularity": -0.02,
    "novelty": 0.05,
    "diversity_adjust": 0.00
  },
  "constraints": {
    "passed": ["exclude_genres", "year_range"],
    "failed": []
  },
  "evidence": {
    "matched_genres": ["Comedy", "Drama"],
    "year": 2002,
    "popularity_quantile": 0.35
  }
}
```

### 7.2 Natural-language explanation (LLM; evidence-bound)
**Inputs to the LLM:**
- Control JSON for this turn
- The structured evidence object (above)
- Optional: short structured profile summary

**Outputs:**
- 2–3 sentences explaining *why this item appears*
- Optional 1 sentence describing a counterfactual (“if you raise novelty …”)

**Hard constraint for faithfulness:**
- The explanation may only cite fields present in `evidence` / `components` / `constraints`.  
  Mentioning unsupported attributes (e.g., “director”, “runtime” when not present) counts as hallucination.

---

## 8) Audit Logging (Replay)

Write a JSONL record per interaction:
- user_id
- timestamp
- user_message
- model_name (LLM under test)
- control_json (raw)
- candidate_ids (Top-N from base)
- filtered_candidate_count
- final_topk (with score breakdown)
- per-item evidence objects
- final natural-language explanations

Goal: *one command* can replay and re-render a session deterministically.

---

## 9) ControlBench: Benchmark for Cross-LLM Comparison

### 9.1 Why ControlBench exists
Most existing benchmarks evaluate end-to-end “chat recommenders.” UGRP instead isolates:
- the LLM’s **control parsing** ability
- the system’s **control execution**
- the LLM’s **faithful explanation rendering**

### 9.2 Task decomposition
- **T1: Intent → Control JSON**
- **T2: Control effectiveness** (does ranking change as intended?)
- **T3: Explanation faithfulness** (does explanation match evidence?)

### 9.3 Automatic test-case generation (MovieLens-based)
Each sample includes:
```json
{
  "user_id": "u_42",
  "history_summary": { "...": "..." },
  "user_message": "Give me 90s crime/mystery, no horror, and slightly more long-tail.",
  "ground_truth_control": { "...": "..." },
  "eval_targets": {
    "must_exclude_genres": ["Horror"],
    "year_min": 1990,
    "year_max": 1999,
    "popularity_bias": -0.2
  }
}
```

Generation approach:
- Build user profile stats from history (top genres, year distribution, popularity bias)
- Sample 1–2 constraint dimensions per test (genre/year/novelty/diversity/popularity)
- Produce paraphrases (synonyms, colloquial phrasing, noise injection) to test robustness

Target volume: **5k–10k** samples for v0.1.

---

## 10) Evaluation Metrics and Leaderboard

### 10.1 T1 (parser) metrics
- **Schema validity rate**: valid JSON w/ required keys + correct types
- **Constraint extraction F1**: for genre/year constraints vs ground truth
- **Continuous-value error (MAE)**: novelty/diversity/popularity_bias
- **Stability**: repeated runs agreement (temperature fixed; or multiple seeds)

### 10.2 T2 (control effectiveness) metrics
- **Constraint satisfaction rate**: percentage of outputs that satisfy hard constraints
- **Monotonicity tests**:
  - novelty ↑ ⇒ average popularity quantile ↓
  - diversity ↑ ⇒ genre coverage / intra-list similarity improves
- **Ranking shift magnitude**:
  - Top-K overlap
  - Kendall tau distance (optional)

### 10.3 T3 (explanation faithfulness) metrics
- **Hallucination rate**: mentions of attributes not present in evidence
- **Evidence coverage**: whether explanation references top contributing factors
- **Counterfactual correctness (optional)**: stated direction matches actual changes

### 10.4 Aggregate score (recommended)
```
UGRP Score = 0.40 * T1 + 0.35 * T2 + 0.25 * T3
```
Outputs:
- leaderboard table
- bar charts (overall + by task)
- radar plot (optional)

---

## 11) Repository Layout and Minimal APIs

### 11.1 Suggested repo structure
```
ugrp/
  data/                 # raw + processed (parquet)
  src/
    profile/            # user profile stats + summaries
    recsys/             # ALS/MF training + candidate generation
    control/            # schema, validator, intent parsing
    rerank/             # deterministic reranker + diversity
    explain/            # evidence builder + LLM renderer
    bench/              # ControlBench generator + evaluators
    adapters/           # LLM adapters (gpt, gemini, ...)
  ui/                   # Streamlit/Gradio/Next.js
  docs/                 # prompts, examples, spec
  outputs/
    logs/               # audit logs (jsonl)
    reports/            # tables + plots
```

### 11.2 Minimal API surface (for UI)
- `GET /user/{id}/profile` → profile card + evidence list
- `POST /chat` (user_id, message, model_name, state) →
  - control_json
  - topk items + score breakdown
  - natural-language explanation
  - audit_id
- `GET /audit/{audit_id}` → full replay bundle

---

## 12) Implementation Milestones (Shortest Path)

### M1 — Base + Profile (1–2 days)
- ingest MovieLens
- train ALS/MF
- Top-200 candidates per user
- profile stats + simple text summary (no LLM needed yet)

### M2 — Schema + Deterministic Reranker (2–3 days)
- Control JSON schema + validator
- hard filtering + soft scoring + diversity greedy/MMR
- structured evidence generation (score breakdown)

### M3 — LLM parser + LLM explanation renderer (2–3 days)
- intent parser: strict JSON output (validate + repair loop)
- explanation renderer: evidence-bound output only
- audit logging (JSONL)

### M4 — ControlBench + Leaderboard (2–4 days)
- generate 5k–10k benchmark cases
- run at least 2 LLMs (e.g., GPT vs Gemini) with identical settings
- output metrics + plots + summary report

### M5 — UI (optional but recommended)
- quick UI in Streamlit/Gradio:
  - profile card + control JSON + recommendation list + chat

---

## 13) v0.1 Deliverables Checklist

- [ ] Trained base recommender and candidate generator (Top-200)
- [ ] Profile builder + profile card (structured + optional LLM narration)
- [ ] Control JSON schema + validator
- [ ] Deterministic reranker (constraints + preferences + diversity)
- [ ] Evidence builder (per-item score breakdown + constraint checks)
- [ ] LLM intent parser (NL → JSON)
- [ ] LLM explanation renderer (evidence-bound)
- [ ] Audit logs (JSONL replay)
- [ ] ControlBench generator (>= 5k cases)
- [ ] Evaluator + leaderboard + plots

---

## 14) Optional v0.2 Extensions (parking lot)
- Add a second dataset (text-heavy): MIND / Amazon Reviews
- Multi-turn state tracking (carry forward control state)
- External metadata enrichment (runtime/director) with stricter evidence contracts
- More nuanced user controls (safety, topic filters, “less of this” feedback)

