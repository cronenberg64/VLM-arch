import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.factory import ModelFactory
from src.benchmark.profiler import Profiler

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Benchmarking {cfg.model.name}...")
    
    model = ModelFactory.create_model(cfg)
    input_size = (1, 3, cfg.dataset.image_size, cfg.dataset.image_size)
    
    device = cfg.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    profiler = Profiler(model, input_size=input_size, device=device)
    
    print("Counting params...")
    params = profiler.count_params()
    print("Counting FLOPs...")
    flops = profiler.count_flops()
    print("Measuring latency...")
    latency = profiler.measure_latency()
    print("Measuring throughput...")
    throughput = profiler.measure_throughput(batch_size=cfg.training.batch_size)
    print("Measuring memory...")
    memory = profiler.measure_peak_memory()
    
    results = {
        "Model": cfg.model.name,
        "Params (M)": params / 1e6,
        "FLOPs (G)": flops / 1e9,
        "Latency (ms)": latency,
        "Throughput (img/s)": throughput,
        "Peak Memory (MB)": memory
    }
    
    print("\nResults:")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
        
    # Save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame([results])
    df.to_csv(f"results/benchmark_{cfg.model.name}.csv", index=False)

if __name__ == "__main__":
    main()
