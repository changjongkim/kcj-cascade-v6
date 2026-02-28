import glob
import re
import os

systems = ["cascade", "lmcache_disk", "pdc", "llm_gpu", "hdf5_indep", "redis"]
results = {}

for sys in systems:
    logs = glob.glob(f"benchmark/logs/v9_{sys}_*.out")
    if not logs: 
        logs = glob.glob(f"benchmark/logs/v9_{sys.upper()}_*.out") + glob.glob(f"benchmark/logs/v9_{sys.lower()}_*.out")
    
    latest_log = None
    max_time = 0
    for l in logs:
        mtime = os.path.getmtime(l)
        if mtime > max_time:
            max_time = mtime
            latest_log = l
    
    if latest_log:
        with open(latest_log, "r") as f:
            content = f.read()
            blocks = re.finditer(r"Benchmark Results \((\d+) Nodes\).+?Avg TTFT[^:]*:\s*([\d\.]+) ms.+?Throughput([^:]*):\s*([\d\.]+)", content, re.DOTALL)
            for b in blocks:
                nodes = int(b.group(1))
                ttft = float(b.group(2))
                throughput_type = b.group(3)
                tput = float(b.group(4))
                
                # Check if it is aggregate or not based on the label 
                # (the script we updated prints "Aggregate Throughput")
                is_agg = ("Aggregate" in throughput_type)
                
                if not is_agg:
                    tput = tput * nodes
                    
                if sys not in results:
                    results[sys] = {}
                results[sys][nodes] = {"ttft": ttft, "tput": tput}

for sys, data in results.items():
    print(f"System: {sys}")
    for n in sorted(data.keys()):
        print(f"  {n} Nodes - TTFT: {data[n]['ttft']} ms, Throughput: {data[n]['tput']} req/s")
