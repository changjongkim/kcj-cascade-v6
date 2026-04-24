
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

TOTAL_REQUESTS = 128

def generate_prompts(num_prompts: int, offset: int = 0,
                     target_tokens: int = 0,
                     rank_tag: int = -1) -> list[str]:
    if rank_tag >= 0:
        shared_prefix = (
            f"[Node {rank_tag} context] You are a helpful AI assistant. "
            f"Please answer the following question about distributed computing systems. "
        )
    else:
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

    padding_sentences = [
        "Distributed systems require careful consideration of network partitions and failures. ",
        "The trade-offs between consistency and availability are fundamental to system design. ",
        "Replication strategies must account for both synchronous and asynchronous approaches. ",
        "Load balancing across multiple nodes improves overall system throughput significantly. ",
        "Fault tolerance mechanisms ensure the system continues operating despite component failures. ",
        "Data partitioning schemes determine how information is distributed across the cluster. ",
        "Consensus protocols enable multiple nodes to agree on a single value reliably. ",
        "Cache coherence protocols maintain data consistency across distributed memory hierarchies. ",
    ]

    prompts = []
    for i in range(num_prompts):
        idx = (offset + i) % len(topics)
        base = shared_prefix + topics[idx]
        if target_tokens > 0:
            target_chars = target_tokens * 4
            pad_idx = 0
            while len(base) < target_chars:
                base += padding_sentences[pad_idx % len(padding_sentences)]
                pad_idx += 1
        prompts.append(base)
    return prompts

def measure_inference(llm, prompts, sampling_params, label):
    from vllm import SamplingParams

    print(f"\n--- {label} ---")
    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.time()

    total_time = t_end - t_start
    total_input = sum(len(o.prompt_token_ids) for o in outputs)
    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)

    throughput_in = total_input / total_time if total_time > 0 else 0
    throughput_out = total_output / total_time if total_time > 0 else 0
    avg_ttft = (total_time / len(prompts)) * 1000

    result = {
        "label": label,
        "num_requests": len(prompts),
        "total_time_s": total_time,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "throughput_input_toks_s": throughput_in,
        "throughput_output_toks_s": throughput_out,
        "throughput_total_toks_s": (total_input + total_output) / total_time if total_time > 0 else 0,
        "avg_ttft_ms": avg_ttft,
    }

    print(f"  Requests: {len(prompts)}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Input throughput:  {throughput_in:.1f} toks/s")
    print(f"  Output throughput: {throughput_out:.1f} toks/s")
    print(f"  Avg TTFT: {avg_ttft:.2f} ms/req")

    return result

def main():
    parser = argparse.ArgumentParser(description="Pure vLLM Baseline Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2112)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--target-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    requests_per_rank = TOTAL_REQUESTS // world_size
    prompt_offset = rank * requests_per_rank

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Pure vLLM Baseline — {world_size}N")
    print(f"{'='*60}")
    print(f"Job:      {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Rank:     {rank}/{world_size}")
    print(f"Model:    {args.model}")
    print(f"Requests: {TOTAL_REQUESTS} total, {requests_per_rank} per rank")
    print(f"{'='*60}")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
    )
    model_load_s = time.time() - t0
    print(f"  Model load: {model_load_s:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=args.max_tokens,
    )

    t0 = time.time()
    _ = llm.generate(["Hello world"], SamplingParams(max_tokens=1))
    warmup_s = time.time() - t0
    print(f"  Warmup: {warmup_s:.2f}s")

    prompts = generate_prompts(requests_per_rank, offset=prompt_offset,
                               target_tokens=args.target_tokens,
                               rank_tag=rank)
    t0 = time.time()
    pass1 = measure_inference(llm, prompts, sampling_params, "Pass 1: Compute (no Cascade)")
    pass1_s = time.time() - t0

    if world_size > 1:
        other_rank = (rank + 1) % world_size
        other_offset = other_rank * requests_per_rank
        cross_prompts = generate_prompts(
            requests_per_rank, offset=other_offset,
            target_tokens=args.target_tokens,
            rank_tag=other_rank,
        )
    else:
        cross_prompts = prompts

    t0 = time.time()
    pass2 = measure_inference(llm, cross_prompts, sampling_params, "Pass 2: Prefix Cache Hit")
    pass2_s = time.time() - t0

    speedup = pass1["total_time_s"] / pass2["total_time_s"] if pass2["total_time_s"] > 0 else 0

    result = {
        "rank": rank,
        "world_size": world_size,
        "model": args.model,
        "total_requests": TOTAL_REQUESTS,
        "requests_this_rank": requests_per_rank,
        "max_tokens": args.max_tokens,
        "stage_times": {
            "model_load_s": model_load_s,
            "warmup_s": warmup_s,
            "pass1_s": pass1_s,
            "pass2_s": pass2_s,
        },
        "pass1": pass1,
        "pass2": pass2,
        "speedup": speedup,
    }

    print(f"\n[Rank {rank}] Summary:")
    print(f"  Pass 1: {pass1['total_time_s']:.3f}s ({pass1['throughput_output_toks_s']:.1f} out toks/s)")
    print(f"  Pass 2: {pass2['total_time_s']:.3f}s ({pass2['throughput_output_toks_s']:.1f} out toks/s)")
    print(f"  Speedup: {speedup:.2f}x")

    rank_path = Path(args.output_dir) / f"rank_{rank}.json"
    with open(rank_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nRank {rank} results saved to {rank_path}")

    if rank == 0:

        barrier_dir = Path(args.output_dir) / "_barrier"
        barrier_dir.mkdir(exist_ok=True)
        (barrier_dir / f"done_{rank}").touch()
        for r in range(world_size):
            deadline = time.time() + 300
            while not (barrier_dir / f"done_{r}").exists():
                if time.time() > deadline:
                    break
                time.sleep(1)

        all_results = []
        for r in range(world_size):
            p = Path(args.output_dir) / f"rank_{r}.json"
            if p.exists():
                with open(p) as f:
                    all_results.append(json.load(f))

        if all_results:
            p1_times = [r["pass1"]["total_time_s"] for r in all_results]
            p2_times = [r["pass2"]["total_time_s"] for r in all_results]
            p1_wall = max(p1_times)
            p2_wall = max(p2_times)
            p1_thr = sum(r["pass1"]["throughput_output_toks_s"] for r in all_results)
            p2_thr = sum(r["pass2"]["throughput_output_toks_s"] for r in all_results)

            aggregate = {
                "nodes": world_size,
                "total_requests": TOTAL_REQUESTS,
                "pass1": {"wall_time_s": p1_wall, "aggregate_throughput_toks_s": p1_thr},
                "pass2": {"wall_time_s": p2_wall, "aggregate_throughput_toks_s": p2_thr},
                "speedup": p1_wall / p2_wall if p2_wall > 0 else 0,
            }
            agg_path = Path(args.output_dir) / "aggregate.json"
            with open(agg_path, "w") as f:
                json.dump(aggregate, f, indent=2)

            print(f"\n{'='*60}")
            print(f"BASELINE RESULTS — {world_size} Node(s)")
            print(f"{'='*60}")
            print(f"Pass 1: {p1_wall:.3f}s  throughput: {p1_thr:.1f} toks/s")
            print(f"Pass 2: {p2_wall:.3f}s  throughput: {p2_thr:.1f} toks/s")
            print(f"Speedup: {aggregate['speedup']:.2f}x")
            print(f"{'='*60}")
    else:
        barrier_dir = Path(args.output_dir) / "_barrier"
        barrier_dir.mkdir(exist_ok=True)
        (barrier_dir / f"done_{rank}").touch()

if __name__ == "__main__":
    main()
