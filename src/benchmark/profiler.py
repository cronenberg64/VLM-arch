import torch
import time
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count
import psutil
import os

class Profiler:
    def __init__(self, model, input_size=(1, 3, 224, 224), device='cuda'):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.model.to(device)
        self.model.eval()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def count_flops(self):
        dummy_input = torch.randn(self.input_size).to(self.device)
        flops = FlopCountAnalysis(self.model, dummy_input)
        return flops.total()

    def measure_latency(self, num_runs=100, warmup=10):
        dummy_input = torch.randn(self.input_size).to(self.device)
        
        # Warmup
        for _ in range(warmup):
            _ = self.model(dummy_input)
            
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model(dummy_input)
            
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_latency = (end_time - start_time) / num_runs
        return avg_latency * 1000  # ms

    def measure_throughput(self, batch_size=32, num_runs=50):
        dummy_input = torch.randn(batch_size, *self.input_size[1:]).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
            
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model(dummy_input)
            
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        total_time = end_time - start_time
        throughput = (batch_size * num_runs) / total_time
        return throughput # images/sec

    def measure_peak_memory(self):
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randn(self.input_size).to(self.device)
            _ = self.model(dummy_input)
            return torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        else:
            return 0.0 # Difficult to measure precise peak CPU memory in python without external tools
