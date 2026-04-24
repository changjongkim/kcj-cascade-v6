
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def run_with_cascade(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    storage_path: str,
    gpu_memory_utilization: float = 0.85,
) -> dict:

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig

    print(f"\n{'='*60}")
    print(f"Starting vLLM with Cascade KV connector")
    print(f"Model: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Storage: {storage_path}")
    print(f"{'='*60}\n")

    kv_transfer_config = KVTransferConfig(
        kv_connector="CascadeConnectorV1",
        kv_connector_module_path="cascade_connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "storage_path": storage_path,
            "dram_capacity_gb": 32.0,
            "enable_dedup": True,
            "enable_compression": True,
        },
    )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        kv_transfer_config=kv_transfer_config,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    print(f"\n--- Pass 1: Computing + Storing KV cache ---")
    start = time.time()
    outputs_1 = llm.generate(prompts, sampling_params)
    pass1_time = time.time() - start

    pass1_results = []
    for output in outputs_1:
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        pass1_results.append({"tokens": tokens, "text_preview": text[:80]})

    print(f"Pass 1 time: {pass1_time:.2f}s")
    print(f"Total tokens: {sum(r['tokens'] for r in pass1_results)}")

    print(f"\n--- Pass 2: Reusing Cascade KV cache ---")
    start = time.time()
    outputs_2 = llm.generate(prompts, sampling_params)
    pass2_time = time.time() - start

    pass2_results = []
    for output in outputs_2:
        text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        pass2_results.append({"tokens": tokens, "text_preview": text[:80]})

    print(f"Pass 2 time: {pass2_time:.2f}s")
    print(f"Total tokens: {sum(r['tokens'] for r in pass2_results)}")

    speedup = pass1_time / pass2_time if pass2_time > 0 else 0
    print(f"\n  Speedup from cache: {speedup:.2f}x")

    return {
        "model": model_name,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "storage_path": storage_path,
        "pass1_time_s": pass1_time,
        "pass2_time_s": pass2_time,
        "speedup": speedup,
        "pass1_results": pass1_results,
        "pass2_results": pass2_results,
    }

def run_baseline(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    gpu_memory_utilization: float = 0.85,
) -> dict:

    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Baseline: vLLM without Cascade")
    print(f"{'='*60}\n")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    baseline_time = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Baseline time: {baseline_time:.2f}s")
    print(f"Total tokens: {total_tokens}")

    return {
        "baseline_time_s": baseline_time,
        "total_tokens": total_tokens,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Cascade+vLLM End-to-End Integration Test"
    )
    parser.add_argument(
        "--model", type=str, default="facebook/opt-125m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=8,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--storage-path", type=str,
        default="${SCRATCH}/cascade_kv_store",
        help="Lustre path for Cascade KV storage",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline comparison",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.85,
        help="GPU memory fraction for vLLM",
    )
    args = parser.parse_args()

    shared_prefix = (
        "You are a helpful AI assistant. Please answer the following "
        "question about distributed computing systems. "
    )
    topics = [
        "What is the CAP theorem and why does it matter?",
        "Explain consensus algorithms like Paxos and Raft.",
        "How does MapReduce work for large-scale data processing?",
        "What are the trade-offs of strong vs eventual consistency?",
        "Describe the architecture of a distributed key-value store.",
        "How does sharding improve database scalability?",
        "What is the two-phase commit protocol?",
        "Explain vector clocks in distributed systems.",
        "How do CRDTs enable conflict-free replication?",
        "What are the challenges of distributed transaction processing?",
        "Describe how Raft leader election works.",
        "What is gossip protocol and when is it used?",
        "Explain the difference between CP and AP systems.",
        "How does consistent hashing help with load balancing?",
        "What are anti-entropy protocols?",
        "Describe the Dynamo architecture.",
    ]
    prompts = [
        shared_prefix + topic
        for topic in topics[:args.num_prompts]
    ]

    rank = int(os.environ.get("SLURM_PROCID", 0))
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    print(f"{'='*60}")
    print(f"Cascade+vLLM E2E Test")
    print(f"{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Rank: {rank}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Storage: {args.storage_path}")
    print(f"{'='*60}")

    results = {}

    try:
        results["cascade"] = run_with_cascade(
            model_name=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            storage_path=args.storage_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except Exception as e:
        print(f"\nERROR running Cascade: {e}")
        import traceback
        traceback.print_exc()
        results["cascade_error"] = str(e)

    if not args.skip_baseline:
        try:
            results["baseline"] = run_baseline(
                model_name=args.model,
                prompts=prompts,
                max_tokens=args.max_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        except Exception as e:
            print(f"\nERROR running baseline: {e}")
            results["baseline_error"] = str(e)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if "cascade"in results:
        c = results["cascade"]
        print(f"Cascade Pass 1 (compute+store): {c['pass1_time_s']:.2f}s")
        print(f"Cascade Pass 2 (cache hit):     {c['pass2_time_s']:.2f}s")
        print(f"Cache speedup:                  {c['speedup']:.2f}x")
    if "baseline"in results:
        b = results["baseline"]
        print(f"Baseline (no Cascade):          {b['baseline_time_s']:.2f}s")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"cascade_e2e_rank{rank}_{job_id}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
