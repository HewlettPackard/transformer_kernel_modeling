This folder includes Llama-7B model (based on lit-llama) pretraing benchmarking results on different GPUs with different batch sizes using FSDP. 

- Batch size: `1, 2, 4, 8` 
- GPU server: 8 * `V100-SXM2-32GB`, 8 * `A100-SXM4-80GB`, 4 * `H100-PCIe-80GB`

For the FSDP strategy, the following configuration is used.
```python
strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True
)
```

Missing files indicates that benchmarking was not completed successfully due to `OOM` error.
