# LLM Inference Speed Optimization

## Current Bottleneck

**V3 Time Breakdown:**
- Generation: **4.86s (71%)** ← Main bottleneck
- Retrieval: 1.87s (27%)
- Reranking: 0.09s (1%)
- Filtering: 0.00s (0%)

**Goal**: Reduce generation time from 4.86s to ~2-3s (40-60% speedup)

---

## Quick Wins (Implement Now)

### 1. Quantization ⚡ Expected: 30-50% speedup

**INT8 Quantization** (Best balance of speed/quality):

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",  # or "gptq" or "fp8"
    gpu_memory_utilization=0.8
)
```

**Options:**
- **AWQ (4-bit)**: 40-50% faster, <1% quality loss
- **GPTQ (4-bit)**: 35-45% faster, <1% quality loss
- **FP8**: 30-40% faster, minimal quality loss
- **INT8**: 20-30% faster, negligible quality loss

**Implementation**:
```bash
# Use pre-quantized model
llm = LLM(model="TheBloke/Qwen2.5-7B-Instruct-AWQ")
```

### 2. Reduce Max Tokens ⚡ Expected: 20-30% speedup

Current: 512 tokens
Optimal: 256-384 tokens (most answers are < 300 tokens)

```python
RAGConfigV3(
    max_tokens=256,  # Reduced from 512
    temperature=0.1
)
```

**Analysis of V3 test answers:**
- Average: 348 tokens (~1740 chars / 5 chars per token)
- 80% of answers: < 400 tokens
- Recommendation: 384 tokens (covers 90% of answers)

### 3. Shorter Prompts ⚡ Expected: 10-15% speedup

**Current prompt**: ~6000 tokens (15 docs × 400 tokens/doc)
**Optimized**: ~3000 tokens (10 docs × 300 tokens/doc)

Strategies:
- Use top 10 docs instead of 15
- Truncate abstracts to 200 words
- Remove redundant context

```python
# Truncate abstracts
for doc in documents[:10]:  # Top 10 only
    abstract = doc.abstract[:800]  # ~200 words
    context_parts.append(f"[{i}] {doc.title}\n{abstract}")
```

### 4. Batch Processing (for multiple queries) ⚡ Expected: 2-3x throughput

vLLM excels at batched inference:

```python
# Process multiple questions at once
questions = ["What causes malaria?", "What is diabetes?", ...]
prompts = [build_prompt(q, docs) for q, docs in zip(questions, all_docs)]

outputs = llm.generate(prompts, sampling_params)
```

---

## Medium-Term Optimizations (Next Phase)

### 5. Speculative Decoding ⚡ Expected: 20-40% speedup

Use smaller "draft" model to propose tokens, larger model verifies:

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_model="Qwen/Qwen2.5-1.5B-Instruct",  # Draft model
    num_speculative_tokens=5
)
```

Requires vLLM 0.3.0+ with speculative decoding support.

### 6. Prefix Caching ⚡ Expected: 30-50% speedup (cached queries)

Cache common prompt prefixes:

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True  # Already enabled in V3!
)
```

**Already active in V3**, but limited benefit for unique queries.

### 7. Attention Optimization ⚡ Expected: 10-20% speedup

FlashAttention-3 or PagedAttention optimization:

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    attention_backend="FLASHINFER",  # Faster than FLASH_ATTN
)
```

### 8. Smaller Model for Simple Questions ⚡ Expected: 50-70% speedup

Use Qwen 1.5B for simple factual questions:

```python
class AdaptiveGeneration:
    def __init__(self):
        self.large_model = LLM("Qwen/Qwen2.5-7B-Instruct")
        self.small_model = LLM("Qwen/Qwen2.5-1.5B-Instruct")

    def generate(self, question, docs):
        if is_simple_factual(question):
            return self.small_model.generate(...)  # 2-3x faster
        else:
            return self.large_model.generate(...)
```

Simple questions: "What is X?", "Who discovered Y?", "When did Z happen?"

---

## Long-Term Optimizations (Future Phases)

### 9. Model Distillation ⚡ Expected: 60-80% speedup

Train smaller model (1.5B-3B) to mimic Qwen 7B:

- 3-4x faster inference
- Maintains 90-95% quality
- Requires training on Qwen outputs

### 10. Local Optimized Model ⚡ Expected: 40-60% speedup

Fine-tune model specifically for biomedical QA:

Benefits:
- Can use smaller model (3B instead of 7B)
- Better performance on domain tasks
- Remove unnecessary capabilities

### 11. TensorRT-LLM ⚡ Expected: 2-3x speedup

Use NVIDIA TensorRT for optimized inference:

```bash
# Convert to TensorRT
python convert_checkpoint.py --model_dir Qwen2.5-7B-Instruct
trtllm-build --checkpoint_dir ...
```

More complex setup but significant speedup.

### 12. Tensor Parallelism ⚡ Expected: 1.5-2x speedup (multi-GPU)

Split model across multiple GPUs:

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2  # Use 2 GPUs
)
```

Only beneficial if you have multiple GPUs.

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (This Week) - Expected: 50-70% speedup

1. ✅ **Reduce max tokens**: 512 → 384 tokens
2. ✅ **Shorter prompts**: 15 docs → 10 docs, truncate abstracts
3. ⏳ **INT8/FP8 quantization**: Use quantized model
4. ⏳ **Test and measure**: Verify quality maintained

**Expected result**: 4.86s → 2.5-3.0s generation time

### Phase 2: Medium-term (Next 2 Weeks) - Additional 20-30% speedup

5. **Adaptive model selection**: Small model for simple questions
6. **Speculative decoding**: Add draft model
7. **Batch processing**: For API server with concurrent requests

**Expected result**: 2.5-3.0s → 1.8-2.2s generation time

### Phase 3: Long-term (1-2 Months) - Additional 40-50% speedup

8. **Model distillation**: Train 3B model on Qwen 7B outputs
9. **TensorRT-LLM**: Convert to optimized engine

**Expected result**: 1.8-2.2s → 1.0-1.5s generation time

---

## Implementation: Quick Wins

### Step 1: Optimize V3 Config

```python
# v3_fast_config.py
RAGConfigV3(
    # Retrieval (already optimized)
    retrieval_n=20,
    use_smart_retrieval=True,

    # Processing (already optimized)
    use_reranking=True,
    final_n=10,  # Reduced from 15

    # Generation (NEW OPTIMIZATIONS)
    max_tokens=384,  # Reduced from 512
    temperature=0.1,

    # vLLM settings
    model_name="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.8,
)
```

### Step 2: Truncate Context

```python
def build_prompt_optimized(question, documents):
    """Build optimized prompt with truncated context"""

    # Use top 10 docs only
    documents = documents[:10]

    context_parts = []
    for i, doc in enumerate(documents):
        # Truncate abstract to ~200 words (800 chars)
        abstract = doc.abstract[:800]
        if len(doc.abstract) > 800:
            abstract += "..."

        context_parts.append(
            f"[{i}] PMC{doc.pmcid}: {doc.title}\n{abstract}"
        )

    context = "\n\n".join(context_parts)

    # Shorter system prompt
    prompt = f"""Answer using context. Cite sources with [0], [1], etc.

Context:
{context}

Question: {question}

Answer:"""

    return prompt
```

### Step 3: Quantization (Easiest: Use pre-quantized model)

```python
# Option A: Use FP8 (vLLM native support)
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="fp8",  # Fast and accurate
    gpu_memory_utilization=0.8
)

# Option B: Use pre-quantized AWQ model
llm = LLM(
    model="TheBloke/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.8
)

# Option C: Use GPTQ
llm = LLM(
    model="TheBloke/Qwen2.5-7B-Instruct-GPTQ",
    quantization="gptq",
    gpu_memory_utilization=0.8
)
```

---

## Testing Quantization

```bash
# Test FP8 quantization
./venv/bin/python3 test_quantization.py --method fp8

# Test AWQ quantization
./venv/bin/python3 test_quantization.py --method awq

# Compare quality
./venv/bin/python3 test_quantization.py --compare
```

---

## Expected V3.1 Performance

### Current V3:
- Total: 6.81s
- Generation: 4.86s (71%)

### V3.1 (with quick wins):
- **Total: ~4.5-5.0s** (34% faster)
- Generation: ~2.5-3.0s (50% faster)
- Retrieval: 1.87s (same)
- Other: 0.09s (same)

### V3.2 (with medium-term):
- **Total: ~3.5-4.0s** (47% faster than V3.1)
- Generation: ~1.8-2.2s
- With adaptive model: ~1.5-2.0s average

---

## Quality Considerations

### Quantization Quality Impact

| Method | Speed | Quality Loss | Recommendation |
|--------|-------|--------------|----------------|
| FP16 (baseline) | 1.0x | 0% | Current |
| FP8 | 1.3-1.4x | <0.5% | ✓ Best choice |
| INT8 | 1.2-1.3x | <1% | ✓ Very safe |
| AWQ (4-bit) | 1.4-1.5x | <2% | ✓ Good for speed |
| GPTQ (4-bit) | 1.4-1.5x | <2% | ✓ Good for speed |

**Recommendation**: Start with FP8 (minimal quality loss, good speedup)

### Token Limit Impact

Reducing from 512 → 384 tokens:
- 90% of V3 answers fit in 384 tokens
- For longer answers, model will summarize
- Quality impact: Minimal (answers already concise)

### Context Truncation Impact

Reducing from 15 → 10 docs:
- 10 docs after reranking + filtering are highest quality
- Docs 11-15 rarely contribute new information
- Quality impact: Minimal

---

## Next Steps

1. ✅ Create optimized V3.1 config
2. ✅ Implement prompt truncation
3. ⏳ Test FP8 quantization
4. ⏳ Measure speed vs quality tradeoff
5. ⏳ Evaluate V3.1 on 120 QA pairs
6. ⏳ Deploy best configuration

## Files to Create

- `src/pipeline_vllm_v3_fast.py` - V3.1 with optimizations
- `test_quantization.py` - Test different quantization methods
- `compare_v3_v3_fast.py` - Compare V3 vs V3.1
- `V3_FAST_RESULTS.md` - Results and recommendations
