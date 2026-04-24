
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "${REPO_ROOT}/ml_workspace/build_ml_311")

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

def measure_inference(
    llm, prompts: list[str], sampling_params, label: str
) -> dict:
    from vllm import SamplingParams

    print(f"\n--- {label} ---")
    request_metrics = []

    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_end = time.time()

    total_time = t_end - t_start
    total_input_tokens = 0
    total_output_tokens = 0

    for i, output in enumerate(outputs):
        num_input = len(output.prompt_token_ids)
        num_output = len(output.outputs[0].token_ids)
        total_input_tokens += num_input
        total_output_tokens += num_output

        request_metrics.append({
            "request_id": i,
            "input_tokens": num_input,
            "output_tokens": num_output,
            "text_preview": output.outputs[0].text[:60],
        })

    throughput_input = total_input_tokens / total_time if total_time > 0 else 0
    throughput_output = total_output_tokens / total_time if total_time > 0 else 0
    avg_ttft = (total_time / len(prompts)) * 1000

    output_tokens_list = [m["output_tokens"] for m in request_metrics]

    result = {
        "label": label,
        "num_requests": len(prompts),
        "total_time_s": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "throughput_input_toks_s": throughput_input,
        "throughput_output_toks_s": throughput_output,
        "throughput_total_toks_s": (total_input_tokens + total_output_tokens) / total_time if total_time > 0 else 0,
        "avg_ttft_ms": avg_ttft,
        "avg_output_tokens": float(np.mean(output_tokens_list)),
        "requests": request_metrics,
    }

    print(f"Requests: {len(prompts)}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Input throughput:  {throughput_input:.1f} toks/s")
    print(f"Output throughput: {throughput_output:.1f} toks/s")
    print(f"Avg TTFT: {avg_ttft:.2f} ms/req")

    return result

def run_rank(
    model_name: str,
    max_tokens: int,
    storage_path: str,
    rank: int,
    world_size: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    max_num_seqs: int = 256,
    target_tokens: int = 0,
    enforce_eager: bool = False,
) -> dict:

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig

    requests_per_rank = TOTAL_REQUESTS // world_size
    prompt_offset = rank * requests_per_rank
    prompts = generate_prompts(requests_per_rank, offset=prompt_offset,
                               target_tokens=target_tokens,
                               rank_tag=rank)

    print(f"[Rank {rank}/{world_size}] {requests_per_rank} requests "
          f"(offset={prompt_offset}, tp={tensor_parallel_size})")

    stage_times = {}

    t0 = time.time()

    daemon_socket = os.environ.get("CASCADE_DAEMON_SOCKET", "")
    kv_transfer_config = KVTransferConfig(
        kv_connector="CascadeConnectorV1",
        kv_connector_module_path="cascade_connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "storage_path": storage_path,
            "dram_capacity_gb": 32.0,
            "enable_dedup": True,
            "enable_compression": True,
            "daemon_socket": daemon_socket,
        },
    )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        kv_transfer_config=kv_transfer_config,
    )

    stage_times["model_load_s"] = time.time() - t0
    print(f"Model load: {stage_times['model_load_s']:.2f}s")

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    daemon_socket = os.environ.get("CASCADE_DAEMON_SOCKET", "")
    if daemon_socket:
        import socket as _sock
        try:
            s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
            s.connect(daemon_socket)
            s.close()
            print(f"[Rank {rank}] Daemon socket OK before warmup")
        except Exception as e:
            print(f"[Rank {rank}] WARNING: Daemon socket DEAD before warmup: {e}")
    t0 = time.time()
    _ = llm.generate(["Hello world"], SamplingParams(max_tokens=1))
    stage_times["warmup_s"] = time.time() - t0
    print(f"Warmup: {stage_times['warmup_s']:.2f}s")

    t0 = time.time()
    pass1 = measure_inference(llm, prompts, sampling_params, "Pass 1: Compute + Store")
    stage_times["pass1_s"] = time.time() - t0

    daemon_socket = os.environ.get("CASCADE_DAEMON_SOCKET", "")
    if daemon_socket and world_size > 1:
        try:
            from cascade_daemon import CascadeClient
            client = CascadeClient(daemon_socket)
            print(f"[Rank {rank}] Calling sync_metadata (MPI Allgatherv)...", flush=True)
            client.sync_metadata()
            print(f"[Rank {rank}] sync_metadata done — global index updated", flush=True)
            client.close()
        except Exception as e:
            print(f"[Rank {rank}] WARNING: sync_metadata failed: {e}", flush=True)

    if world_size > 1:

        other_rank = (rank + 1) % world_size
        other_offset = other_rank * requests_per_rank
        cross_prompts = generate_prompts(
            requests_per_rank, offset=other_offset,
            target_tokens=target_tokens,
            rank_tag=other_rank,
        )
        print(f"[Rank {rank}] Pass 2 uses rank {other_rank}'s prompts "
              f"(rank_tag={other_rank}, forces RDMA cross-node fetch)")
    else:

        cross_prompts = prompts

    t0 = time.time()
    pass2 = measure_inference(
        llm, cross_prompts, sampling_params,
        "Pass 2: Cross-Node Cache Hit (RDMA)"
        if world_size > 1 else "Pass 2: Local Cache Hit",
    )
    stage_times["pass2_s"] = time.time() - t0

    speedup = pass1["total_time_s"] / pass2["total_time_s"] if pass2["total_time_s"] > 0 else 0

    result = {
        "rank": rank,
        "world_size": world_size,
        "model": model_name,
        "total_requests": TOTAL_REQUESTS,
        "requests_this_rank": requests_per_rank,
        "max_tokens": max_tokens,
        "stage_times": stage_times,
        "pass1": pass1,
        "pass2": pass2,
        "speedup": speedup,
    }

    print(f"\n[Rank {rank}] Summary:")
    print(f"Pass 1: {pass1['total_time_s']:.3f}s "
          f"({pass1['throughput_output_toks_s']:.1f} out toks/s)")
    print(f"Pass 2: {pass2['total_time_s']:.3f}s "
          f"({pass2['throughput_output_toks_s']:.1f} out toks/s)")
    print(f"Cache speedup: {speedup:.2f}x")

    return result

def aggregate_results(output_dir: str, world_size: int):
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    all_results = []
    for rank in range(world_size):
        path = Path(output_dir) / f"rank_{rank}.json"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        with open(path) as f:
            all_results.append(json.load(f))

    if not all_results:
        print("No results to aggregate")
        return

    p1_times = [r["pass1"]["total_time_s"] for r in all_results]
    p1_throughputs = [r["pass1"]["throughput_output_toks_s"] for r in all_results]
    p1_total_tokens = sum(r["pass1"]["total_output_tokens"] for r in all_results)

    p2_times = [r["pass2"]["total_time_s"] for r in all_results]
    p2_throughputs = [r["pass2"]["throughput_output_toks_s"] for r in all_results]
    p2_total_tokens = sum(r["pass2"]["total_output_tokens"] for r in all_results)

    p1_wall = max(p1_times)
    p2_wall = max(p2_times)

    p1_agg_throughput = sum(p1_throughputs)
    p2_agg_throughput = sum(p2_throughputs)

    p1_ttfts = [r["pass1"]["avg_ttft_ms"] for r in all_results]
    p2_ttfts = [r["pass2"]["avg_ttft_ms"] for r in all_results]

    aggregate = {
        "nodes": world_size,
        "total_requests": TOTAL_REQUESTS,
        "model": all_results[0]["model"],
        "max_tokens": all_results[0]["max_tokens"],
        "pass1": {
            "wall_time_s": p1_wall,
            "total_output_tokens": p1_total_tokens,
            "aggregate_throughput_toks_s": p1_agg_throughput,
            "avg_ttft_ms": float(np.mean(p1_ttfts)),
            "per_rank_times_s": p1_times,
            "per_rank_throughputs_toks_s": p1_throughputs,
        },
        "pass2": {
            "wall_time_s": p2_wall,
            "total_output_tokens": p2_total_tokens,
            "aggregate_throughput_toks_s": p2_agg_throughput,
            "avg_ttft_ms": float(np.mean(p2_ttfts)),
            "per_rank_times_s": p2_times,
            "per_rank_throughputs_toks_s": p2_throughputs,
        },
        "speedup": p1_wall / p2_wall if p2_wall > 0 else 0,
        "per_rank": all_results,
    }

    agg_path = Path(output_dir) / "aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"STRONG SCALING RESULTS — {world_size} Node(s)")
    print(f"{'='*60}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Requests per node: {TOTAL_REQUESTS // world_size}")
    print()
    print(f"Pass 1 (compute+store):")
    print(f"Wall time:   {p1_wall:.3f}s")
    print(f"Throughput:  {p1_agg_throughput:.1f} output toks/s (aggregate)")
    print(f"Avg TTFT:    {np.mean(p1_ttfts):.2f} ms/req")
    print()
    print(f"Pass 2 (cache hit):")
    print(f"Wall time:   {p2_wall:.3f}s")
    print(f"Throughput:  {p2_agg_throughput:.1f} output toks/s (aggregate)")
    print(f"Avg TTFT:    {np.mean(p2_ttfts):.2f} ms/req")
    print()
    print(f"Cache speedup: {aggregate['speedup']:.2f}x")
    print(f"\nSaved to {agg_path}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Cascade+vLLM Strong Scaling Benchmark"
    )
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--storage-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=0,
                        help="Target prompt length in tokens for block size control")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs to save GPU memory")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.storage_path, exist_ok=True)

    print(f"{'='*60}")
    print(f"Cascade+vLLM Strong Scaling — {world_size}N")
    print(f"{'='*60}")
    print(f"Job:      {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Rank:     {rank}/{world_size}")
    print(f"Model:    {args.model}")
    print(f"Requests: {TOTAL_REQUESTS} total, "
          f"{TOTAL_REQUESTS // world_size} per rank")
    print(f"Storage:  {args.storage_path}")
    print(f"{'='*60}")

    if args.aggregate_only:
        aggregate_results(args.output_dir, world_size)
        return

    daemon_proc = None
    daemon_socket = None
    if world_size > 1:
        import multiprocessing as mp

        socket_path = f"/tmp/cascade_daemon_{os.environ.get('SLURM_JOB_ID', 'local')}_{rank}.sock"
        ready_event = mp.Event()

        def _run_daemon(sock_path, storage_path, ready_evt):
            import faulthandler, sys, threading, traceback
            faulthandler.enable(file=sys.stderr)
            try:
                from cascade_daemon import create_daemon
                daemon = create_daemon(
                    lustre_path=storage_path,
                    socket_path=sock_path,
                    dram_capacity_gb=32.0,
                )

                store = daemon._store
                print(f"[Daemon] MPI rank={store.rank} world_size={store.world_size}"
                      f"lustre={storage_path}", flush=True)
                daemon.start()
                daemon.barrier()
                ready_evt.set()

                threading.Event().wait()
            except Exception as e:
                print(f"DAEMON CRASH: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                ready_evt.set()

        daemon_proc = mp.Process(
            target=_run_daemon,
            args=(socket_path, args.storage_path, ready_event),
            daemon=True,
        )
        daemon_proc.start()

        ready_event.wait(timeout=120)
        if not ready_event.is_set():
            raise RuntimeError("Cascade daemon failed to start within 120s")

        daemon_socket = socket_path
        os.environ["CASCADE_DAEMON_SOCKET"] = socket_path
        print(f"[Rank {rank}] Cascade daemon started: {socket_path}")
        print(f"[Rank {rank}] All daemons ready")

    result = run_rank(
        model_name=args.model,
        max_tokens=args.max_tokens,
        storage_path=args.storage_path,
        rank=rank,
        world_size=world_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        target_tokens=args.target_tokens,
        enforce_eager=args.enforce_eager,
    )

    rank_path = Path(args.output_dir) / f"rank_{rank}.json"
    with open(rank_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nRank {rank} results saved to {rank_path}")

    barrier_dir = Path(args.output_dir) / "_barrier"
    barrier_dir.mkdir(exist_ok=True)
    (barrier_dir / f"done_{rank}").touch()
    print(f"[Rank {rank}] Waiting for all ranks to finish...")
    for r in range(world_size):
        deadline = time.time() + 300
        while not (barrier_dir / f"done_{r}").exists():
            if time.time() > deadline:
                print(f"[Rank {rank}] WARNING: Rank {r} did not finish in time")
                break
            time.sleep(1)
    print(f"[Rank {rank}] All ranks done, cleaning up")

    if daemon_proc:
        try:
            from cascade_daemon import CascadeClient
            client = CascadeClient(daemon_socket)
            print(f"\n  [Rank {rank}] Cascade DistributedStore stats:")
            print(f"{client.get_stats()}")
            client.close()
        except Exception as e:
            print(f"[Rank {rank}] Could not get daemon stats: {e}")
        daemon_proc.terminate()
        daemon_proc.join(timeout=5)

    if rank == 0 and world_size > 1:
        aggregate_results(args.output_dir, world_size)
    elif world_size == 1:
        aggregate_results(args.output_dir, world_size)

if __name__ == "__main__":
    main()
