# AIMO3 Competitor Techniques — Deep Research Report

## 1. Competition Technical Framework

### Hardware & Time Budget
- GPU: 1× NVIDIA H100 (80GB VRAM)
- Total notebook runtime: 5 hours (18,000 seconds)
- vLLM loading time: ~10 minutes using OS page cache
- Effective solve time: 4h50m = 17,400 seconds for 50 problems
- Average time per problem: ~348 seconds (5.8 minutes)
- First prediction has no time limit; subsequent predictions must return within 30 minutes

**Sources**: 

### Submission API Pattern
- Module: `kaggle_evaluation.aimo_3_inference_server`
- Must call `inference_server.serve(predict_wrapper)` within 15 minutes of startup
- predict(id_, problem) → DataFrame with [id, answer]
- Answer: non-negative integer, 0 to 99,999 (5 digits)
- Problems served one-by-one in random order
- Internet disabled during scoring
- Gateway runs in separate container

**Sources**: 

### Answer Format
- Each problem specifies its own modulus (mod 10^5, mod 99991, mod 57, etc.)
- Answer must be integer in 
- Critical change from AIMO2 (which was mod 1000, range 0-999)

**Sources**: 

***

## 2. The Dominant Model: GPT-OSS-120B

### Architecture
- 117B total parameters, 5.1B active parameters (Mixture of Experts)
- 128 experts per MoE block, top-4 routing
- MXFP4 quantized out of the box → fits on single 80GB H100
- Residual stream dimension: 2880
- GQA with 64 query heads (dim 64), 8 KV heads
- Alternating banded window (128 tokens) and dense attention
- Context: 131,072 tokens via YaRN scaling
- SwiGLU activation (unconventional implementation with clamping + residual)

**Sources**: 

### Performance Benchmarks (with tools)
- AIME 2024: 96.6% accuracy
- Approaches o4-mini accuracy
- 20K+ CoT tokens per problem average on AIME

### Variable Reasoning Effort
- Three levels: low, medium, high
- Set via system prompt: "Reasoning: high"
- Log-linear test-time scaling: longer CoT = higher accuracy

### Harmony Format (Critical)
- Mandatory prompt format — model does not work without it
- Open-source library: openai/harmony (Python + Rust)
- Channels: analysis (CoT), commentary (tool calls), final (user-visible)
- System prompt must include reasoning level
- In multi-turn: remove reasoning traces from past assistant turns
- Tool calls structured within the format

**Sources**: 

### Running on H100
```
vllm serve openai/gpt-oss-120b
```
- Exposes Chat Completions API on localhost:8000
- Use OpenAI SDK with base_url override
- Supports Agents SDK integration

**Sources**: 

***

## 3. Core Techniques Used by Competitors

### 3.1 Tool-Integrated Reasoning (TIR)

**The single most important technique differentiating top scores.**

Model generates interleaved reasoning text and Python code blocks. Code is executed in a sandbox, output is fed back to the model for continued reasoning.

**Implementation patterns:**

A. Jupyter Kernel (preferred by top teams):
- Persistent state across code blocks within a single problem
- Variables carry over between executions
- Uses jupyter_client library
- More natural for iterative mathematical exploration

B. exec() sandbox (simpler):
- Isolated namespace per code block
- Import sympy, numpy, itertools, math
- 30-second timeout per block
- Capture stdout + variable 'answer'

**Sources**: 

### 3.2 Self-Consistency (Majority Voting)

Generate N solutions (8-16) with temperature=0.7-1.0, extract answers, take most common.

Improvements from AIMO1 winner:
- Weight code-verified answers 0.8 vs text-only 0.05
- Penalize small numbers (0-9)
- Penalize answers appearing in problem text
- Diverse prompts: CoT vs code-first in 3:4 ratio

**Sources**: 

### 3.3 GenSelect (Generative Solution Selection)

NVIDIA innovation: train model to select best solution from N candidates. Better than majority voting, especially with no clear majority. "Self GenSelect" = same model; "32B GenSelect" = dedicated selector.

Not fully integrated in AIMO2 submission due to time constraints.

**Sources**: 

### 3.4 Time Slicing / Dynamic Allocation

- Base time per problem, adjust dynamically
- Early stopping: if 4/5 samples agree, stop
- Bank unused time from easy problems for hard ones
- Emergency fallback if over budget

**Sources**: 

### 3.5 Prompt Engineering Patterns

Multiple templates:
- Chain-of-Thought: pure mathematical reasoning
- Code-First: "Write Python to solve"
- Persona: "You are Terence Tao"
- Topic-specific: combinatorics, number theory, algebra, geometry

**Sources**: 

### 3.6 Model Merging

AIMO2 winner merged CoT and TIR checkpoints: 30% CoT + 70% TIR linear merge via mergekit. Improved both accuracy and reduced solution length.

**Sources**: 

***

## 4. AIMO2 Winner Details (NVIDIA NemoSkills)

### Three Pillars

1. **OpenMathReasoning Dataset**
   - 540K unique problems from AoPS forums
   - 3.2M CoT solutions from DeepSeek-R1 + QwQ-32B
   - Decontaminated against benchmarks
   - Custom validation: Comp-Math-24-25 (AIME + HMMT)

2. **TIR Pipeline**
   - Start with instruction-following model (not reasoning)
   - Fine-tune on limited reasoning data
   - Prompt for TIR solutions (interleaved reasoning + code)
   - Quality filter: code blocks must be "novel and significant"
   - Iterate: better model → more data → repeat
   - Final: 1.7M high-quality TIR solutions

3. **GenSelect**
   - 566K training examples
   - Model compares and evaluates reasoning paths
   - Not just final answers

### Inference
- TensorRT-LLM with FP8/INT8
- ReDrafter speculative decoding
- FastAPI backend, per-problem time budgets (~350s + 200s buffer)
- Early stopping at 4/5 consensus
- Async generation

**Sources**: 

***

## 5. AIMO1 Winner Details (Project Numina)

### Training
- Stage 1: SFT on diverse math dataset (CoT templated)
- Stage 2: TIR fine-tuning (MuMath-Code recipe)
- Stage 3: Reward model via DPO/KTO (sample 4 completions, label pos/neg)
- Key insight: DeepSeekMath-7B-RL as base (multi-step reasoning + code)

### Inference
- Self-Consistency TIR (SC-TIR): 50-140 candidates per problem
- Custom scoring: code execution weight 0.8, text answer 0.05
- Penalty heuristics for small numbers and problem-contained answers
- Parallelized code execution in batches
- Two prompts: CoT + Python/CAS emphasis
- Static KV cache + torch compilation: 2-3x speedup on H100

**Sources**: 

***

## 6. Notable Public Notebooks & Scores

| Notebook | Score | Approach |
|----------|-------|----------|
| datasciencegrad/aimo-3-42-50 | 42/50 | GPT-OSS-120B, Harmony TIR, self-consistency |
| andreasbis/aimo-3-gpt-oss-120b-with-tools | 41/50 | GPT-OSS-120B native tool calling |
| suhaild/aimo3-40-submission | 40/50 | Fork of andreasbis |
| nihilisticneuralnet/gpt-oss-120b-tir | 36-41/50 | TIR + time slicing |
| kaggleqrdl/self-consistency-strategy-tir | ~38-40 | Self-consistency focus |
| seshurajup/aimo-3-gpt-oss-120b-3hours | ~35/50 | Minimal setup |

**Key observation**: ~70% of leaderboard runs variants of same approach.

**Sources**: 

***

## 7. Available Datasets & Resources

- **OpenMathReasoning**: 540K problems, 3.2M solutions 
- **AIMO3 TIR Dataset**: 141,277 TIR training examples from GPT-OSS-120B 
- **AIMO3 Benchmark Dataset**: Community validation set 
- **AIMO3 Reference Problems**: 10 official problems with solutions 
- **AIMO3 Dependency Dataset**: Pre-packaged dependencies for notebooks 

***

## 8. Alternative Models Considered

| Model | Size | AIME Performance | AIMO3 Viability |
|-------|------|-----------------|-----------------|
| gpt-oss-120b | 117B (5.1B active) | 96.6% AIME 2024 | Current LB leader |
| DeepSeek-R1-0528 | 671B MoE | 87.5% AIME 2025 | Too large for single H100 |
| DeepSeek-R1-Distill-Qwen-32B | 32B | ~67% AIME | Fast, good for ensemble |
| QwQ-32B | 32B | ~63% AIME | Good reasoning, code-aware |
| DeepSeek-R1-0528-Qwen3-8B | 8B | Matches 235B-thinking | Distilled, very fast |
| Qwen3-235B-A22B | 235B MoE | ~70% AIME | May fit H100 |

**Sources**: 

***

## 9. Common Failure Modes

1. Index out-of-bounds after first prediction (server API issues)
2. aimo_3_inference_server behaving unpredictably
3. vLLM initialization failures / OOM
4. Code execution timeouts on complex computations
5. Modulus parsing errors (wrong modulus → wrong answer)
6. Answer extraction failures (no \boxed{} found)
7. Time budget overruns (incomplete submission)

**Sources**: 

***

## 10. Gap Analysis: 42/50 → 44+/50

Current top (42) uses: basic self-consistency + TIR + gpt-oss-120b

Missing techniques that could add +2-4 problems:
1. GenSelect-style verification (+1-2)
2. Multi-round TIR with persistent Jupyter state + verification loops (+1)
3. Problem-type classification → specialized prompts (+1)
4. Dynamic model switching (fast model for easy, slow for hard) (+0.5)
5. Robust modulus parsing with cross-validation (+0.5)
6. Aggressive time banking + early stopping optimization (+0.5)