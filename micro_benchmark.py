import itertools
import typing as t
from enum import Enum

import os
import click
import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc

import Kernels as kernels

DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = torch.float16


class Phase(Enum):
    TRAINING_FWD = 0
    TRAINING_BCWD = 1
    INFERENCE_TOKENGEN = 2
    INFERENCE_PREFILL = 3


def benchmark_llama_attn(
    batch,
    seq_len,
    max_seq_len,
    n_embed,
    num_heads,
    tensor_parallelism=1,
    phase=0,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
    warmups=5,
    num_iters=20,
):
    torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True
    )
    timer = kernels.Timer(device)
    opPhase = Phase(phase)

    rope_cache = kernels.build_rope_cache(
        seq_len=max_seq_len,
        n_elem=n_embed // num_heads,
        dtype=dtype,
        device=device
    )
    mask_cache = kernels.build_mask_cache(max_seq_len, device)

    input_pos = None
    kv_cache = None

    if opPhase == Phase.INFERENCE_TOKENGEN:
        assert seq_len == 1
        cache_shape = (
            batch,
            num_heads // tensor_parallelism,
            max_seq_len,
            n_embed // num_heads,
        )
        kv_cache = (
            torch.rand(cache_shape, device=device, dtype=dtype),
            torch.rand(cache_shape, device=device, dtype=dtype),
        )
        input_pos = torch.tensor(
            [max_seq_len // 2], device=device, dtype=int
        )  # torch.arange(0, seq_len, device=device, dtype=int)
        rope_cache = rope_cache.index_select(0, input_pos)
        mask_cache = mask_cache.index_select(2, input_pos)
        mask_cache = mask_cache[:, :, :, :max_seq_len]
    elif opPhase == Phase.INFERENCE_PREFILL:
        assert seq_len == max_seq_len
    else:
        assert seq_len == max_seq_len

    x = torch.randn(
        batch,
        seq_len,
        n_embed,
        device=device,
        dtype=dtype,
        requires_grad=(opPhase == Phase.TRAINING_FWD or opPhase == Phase.TRAINING_BCWD),
    )
    grad = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=False
    )

    llama_config = kernels.LLaMAConfig.from_name("7B")
    llama_config.block_size = max_seq_len
    llama_config.n_head = num_heads
    llama_config.n_embd = n_embed
    llama_config.n_head = num_heads
    llama_config.tensor_parallelism = tensor_parallelism
    llama_config.device = device
    llama_config.dtype = dtype
    causal_attn = kernels.CausalSelfAttention(llama_config)

    if opPhase == Phase.TRAINING_FWD or opPhase == Phase.TRAINING_BCWD:
        for param in causal_attn.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        causal_attn.parameters(),
        lr=6e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        foreach=False,
    )

    causal_attn.to(device)

    times_in_sec = []

    for _ in range(warmups):
        causal_attn(x, rope_cache, mask_cache, max_seq_len, input_pos, kv_cache)
    for _ in range(num_iters):
        y = None
        if opPhase == Phase.TRAINING_BCWD:
            y = causal_attn(x, rope_cache, mask_cache, max_seq_len)[0]
        timer.start()
        if opPhase == Phase.TRAINING_BCWD:
            optimizer.zero_grad()
            y.backward(gradient=grad)
            optimizer.step()
        else:
            causal_attn(x, rope_cache, mask_cache, max_seq_len, input_pos, kv_cache)
        timer.stop()
        # timings are in ms:
        times_in_sec.append(timer.get_latency_ms() / 1e3)
    time_array = np.array(times_in_sec)
    return time_array


def benchmark_llama_mlp(
    batch,
    seq_len,
    n_embed,
    num_heads,
    tensor_parallelism=1,
    phase=0,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
    warmups=5,
    num_iters=20,
):
    timer = kernels.Timer(device)
    inference = False
    if (
        Phase(phase) == Phase.INFERENCE_TOKENGEN
        or Phase(phase) == Phase.INFERENCE_PREFILL
    ):
        inference = True

    x = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=not inference
    )  # requires_grad=True
    grad = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=False
    )
    llama_config = kernels.LLaMAConfig.from_name("7B")
    llama_config.block_size = seq_len
    llama_config.n_head = num_heads
    llama_config.n_embd = n_embed
    llama_config.n_head = num_heads
    llama_config.tensor_parallelism = tensor_parallelism
    llama_config.device = device
    llama_config.dtype = dtype

    mlp = kernels.LLamaMLP(llama_config)
    if not inference:
        for param in mlp.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=6e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        foreach=False,
    )

    mlp.to(device)

    times_in_sec = []

    for _ in range(warmups):
        mlp(x)
    for _ in range(num_iters):
        if Phase(phase) == Phase.TRAINING_BCWD:
            y = mlp(x)
        timer.start()
        if Phase(phase) == Phase.TRAINING_BCWD:
            optimizer.zero_grad()
            y.backward(gradient=grad)
            optimizer.step()
        else:
            mlp(x)
        timer.stop()
        # timings are in ms:
        times_in_sec.append(timer.get_latency_ms() / 1e3)

    time_array = np.array(times_in_sec)
    return time_array


def benchmark_llm_residual_addition(
    batch,
    seq_len,
    n_embed,
    phase=0,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
    warmups=5,
    num_iters=20,
):
    timer = kernels.Timer(device)
    inference = False
    if (
        Phase(phase) == Phase.INFERENCE_TOKENGEN
        or Phase(phase) == Phase.INFERENCE_PREFILL
    ):
        inference = True

    x1 = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=not inference
    )
    x2 = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=not inference
    )
    grad = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=False
    )
    times_in_sec = []

    for _ in range(warmups):
        x1 + x2
    for _ in range(num_iters):
        y = None
        if Phase(phase) == Phase.TRAINING_BCWD:
            y = x1 + x2
        timer.start()
        if Phase(phase) == Phase.TRAINING_BCWD:
            y.backward(gradient=grad)
        else:
            x1 + x2
        timer.stop()
        # timings are in ms:
        times_in_sec.append(timer.get_latency_ms() / 1e3)
    time_array = np.array(times_in_sec)
    return time_array


def benchmark_llm_rmsnorm(
    batch,
    seq_len,
    n_embed,
    phase=0,
    device=DEFAULT_DEVICE,
    dtype=DEFAULT_DTYPE,
    warmups=5,
    num_iters=20,
):
    timer = kernels.Timer(device)
    inference = False
    if (
        Phase(phase) == Phase.INFERENCE_TOKENGEN
        or Phase(phase) == Phase.INFERENCE_PREFILL
    ):
        inference = True

    x = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=not inference
    )
    grad = torch.randn(
        batch, seq_len, n_embed, device=device, dtype=dtype, requires_grad=False
    )
    rms = kernels.RMSNorm(n_embed, dtype)
    rms.to(device)

    times_in_sec = []

    for _ in range(warmups):
        rms(x)
    for _ in range(num_iters):
        y = None
        if Phase(phase) == Phase.TRAINING_BCWD:
            y = rms(x)
        timer.start()
        if Phase(phase) == Phase.TRAINING_BCWD:
            y.backward(gradient=grad)
        else:
            rms(x)
        timer.stop()
        # timings are in ms:
        times_in_sec.append(timer.get_latency_ms() / 1e3)

    time_array = np.array(times_in_sec)
    return time_array


def sobol_sample(
    value_ranges: dict[str, t.Any],
    search_spaces: dict[str, dict[str, list]],
    **kwargs: dict[str, t.Any]
) -> dict[str, t.Any]:
    sampled_experiments = {}

    for k in value_ranges.keys():
        columns: int = len(value_ranges[k])
        sample_exponent: int = kwargs["sample_exponent"]
        sampler = qmc.Sobol(d=columns, scramble=True)

        sample: np.ndarray = sampler.random_base2(m=sample_exponent)

        l_bounds = [value_ranges[k][p][0] for p in value_ranges[k].keys()]
        u_bounds = [value_ranges[k][p][-1] for p in value_ranges[k].keys()]

        # Scale the samples to the defined ranges
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

        # Round to nearest valid values in the search space
        param_keys = list(value_ranges[k].keys())
        adjusted_scaled_sample = np.zeros_like(scaled_sample)

        for i, param in enumerate(param_keys):
            param_values = search_spaces[k][param]
            for j in range(len(scaled_sample)):
                # Find closest value in the parameter space
                adjusted_scaled_sample[j, i] = min(
                    param_values,
                    key=lambda x: abs(x - scaled_sample[j, i])
                )

        sampled_experiments[k] = {
            "normalized_sample": sample,
            "scaled_sample": adjusted_scaled_sample.astype(int),
        }

    return sampled_experiments


def generate_seed_samples(kernel_search_spaces):
    """Generate dataframes with all combinations of parameter values for each kernel."""
    seed_samples = {}
    output_files = []

    for kernel, params in kernel_search_spaces.items():
        # Generate all combinations
        param_names = list(params.keys())
        param_values = [params[param] for param in param_names]
        combinations = list(itertools.product(*param_values))
        df = pd.DataFrame(combinations, columns=param_names)
        seed_samples[kernel] = df

        # Save individual kernel seed sample
        seed_file = f"{kernel}_seed_sample.csv"
        df.to_csv(seed_file, index=False)
        output_files.append(seed_file)

    # Create concatenated version with kernel column
    all_dfs = []

    # Process attn dataframe (rename max_seq_length to seq_length)
    attn_df = seed_samples["attn"].copy()
    attn_df.rename(columns={"max_seq_length": "seq_length"}, inplace=True)
    attn_df["kernel"] = "attn"
    all_dfs.append(attn_df)

    # Process mlp dataframe
    mlp_df = seed_samples["mlp"].copy()
    mlp_df["kernel"] = "mlp"
    all_dfs.append(mlp_df)

    # Get smallest values for missing columns
    min_values = {}
    min_values["n_head"] = min(kernel_search_spaces["attn"]["n_head"])
    min_values["tensor_parallelism"] = min(kernel_search_spaces["attn"]["tensor_parallelism"])

    # Process res_add dataframe
    res_add_df = seed_samples["res_add"].copy()
    res_add_df["kernel"] = "res_add"
    res_add_df["n_head"] = min_values["n_head"]
    res_add_df["tensor_parallelism"] = min_values["tensor_parallelism"]
    all_dfs.append(res_add_df)

    # Process rms_norm dataframe
    rms_norm_df = seed_samples["rms_norm"].copy()
    rms_norm_df["kernel"] = "rms_norm"
    rms_norm_df["n_head"] = min_values["n_head"]
    rms_norm_df["tensor_parallelism"] = min_values["tensor_parallelism"]
    all_dfs.append(rms_norm_df)

    # Concatenate all dataframes
    all_kernels_df = pd.concat(all_dfs, ignore_index=True)

    # Save the concatenated dataframe
    all_seed_file = "all_kernels_seed_sample.csv"
    all_kernels_df.to_csv(all_seed_file, index=False)
    output_files.append(all_seed_file)

    return seed_samples, all_kernels_df, output_files

def build_experiments(
    kernel_search_spaces: dict[str, t.Any],
    sobol_search_spaces: dict[str, t.Any],
    samples: dict[str, t.Any],
    defaults: dict[str, t.Any],
    gpu_search_space: pd.DataFrame,
    seed_samples: dict[str, pd.DataFrame],
    all_seed_df: pd.DataFrame,
    include_sobol: bool = True,
) -> tuple[dict[str, t.Any], list[str]]:
    experiments = {}
    output_files = []

    for k, p in kernel_search_spaces.items():
        # k: kernel names
        # p: dict of parameters
        sobol_p = sobol_search_spaces[k]  # Get expanded parameter space
        p_keys = list(sobol_p.keys())

        experiments[k] = {}
        experiments[k]["scaled_space"] = {}
        experiments[k]["normalized_space"] = {}

        # Only process Sobol samples if include_sobol is True
        if include_sobol and len(samples[k]["scaled_sample"]) > 0:
            for i in range(len(p_keys)):
                param_name = p_keys[i]
                scaled_column = samples[k]["scaled_sample"][:, i]

                experiments[k]["scaled_space"][param_name] = scaled_column
                experiments[k]["normalized_space"][param_name] = samples[k]["normalized_sample"][:, i]

            # Convert to dataframes
            experiments[k]["scaled_space"] = pd.DataFrame(experiments[k]["scaled_space"])
            experiments[k]["normalized_space"] = pd.DataFrame(
                experiments[k]["normalized_space"]
            )
        else:
            # Create empty dataframes if Sobol samples are excluded
            experiments[k]["scaled_space"] = pd.DataFrame(columns=p_keys)
            experiments[k]["normalized_space"] = pd.DataFrame(columns=p_keys)

        # Add GPU parameters and defaults
        for gpu_c in gpu_search_space.columns:
            experiments[k]["scaled_space"][gpu_c] = gpu_search_space[gpu_c].to_numpy()[0]
            experiments[k]["normalized_space"][gpu_c] = gpu_search_space[gpu_c].to_numpy()[0]

        for d_k, d_v in defaults.items():
            experiments[k]["scaled_space"][d_k] = d_v
            experiments[k]["normalized_space"][d_k] = d_v

        # Only append seed samples if they exist
        if seed_samples and k in seed_samples:
            seed_df = seed_samples[k].copy()

            # Add GPU parameters to seed samples
            for gpu_c in gpu_search_space.columns:
                seed_df[gpu_c] = gpu_search_space[gpu_c].to_numpy()[0]

            # Add default parameters
            for d_k, d_v in defaults.items():
                seed_df[d_k] = d_v

            # Append seed samples to scaled space
            experiments[k]["scaled_space"] = pd.concat(
                [experiments[k]["scaled_space"], seed_df],
                ignore_index=True
            )

            # For normalized space, use SOBOL search space for normalization
            normalized_seed_df = seed_df.copy()
            for param_name in p_keys:
                # Get min and max from the EXPANDED sobol search space
                min_val = min(sobol_search_spaces[k][param_name])
                max_val = max(sobol_search_spaces[k][param_name])

                if max_val > min_val:  # Avoid division by zero
                    normalized_seed_df[param_name] = (
                        normalized_seed_df[param_name] - min_val) / (max_val - min_val)
                else:
                    normalized_seed_df[param_name] = 0.0

            experiments[k]["normalized_space"] = pd.concat(
                [experiments[k]["normalized_space"], normalized_seed_df],
                ignore_index=True
            )

    # Create generalist dataframe with all samples
    generalist_scaled_df = []
    generalist_normalized_df = []

    # Only add seed samples to generalist if they exist
    if not all_seed_df.empty:
        # Add the all_seed_df to generalist dataframe
        all_seed_with_gpu = all_seed_df.copy()
        for gpu_c in gpu_search_space.columns:
            all_seed_with_gpu[gpu_c] = gpu_search_space[gpu_c].to_numpy()[0]
        for d_k, d_v in defaults.items():
            all_seed_with_gpu[d_k] = d_v
        generalist_scaled_df.append(all_seed_with_gpu)

        # Normalize the seed values for the normalized generalist dataframe
        # using kernel search spaces directly
        all_seed_normalized = all_seed_with_gpu.copy()
        for kernel_name, params in kernel_search_spaces.items():
            kernel_mask = all_seed_normalized["kernel"] == kernel_name

            for param_name, param_values in params.items():
                # Handle the special case of max_seq_length vs seq_length
                col_name = "seq_length" if param_name == "max_seq_length" else param_name

                if col_name in all_seed_normalized.columns:
                    min_val = min(param_values)
                    max_val = max(param_values)

                    if max_val > min_val:
                        # Apply normalization only to rows of this specific kernel
                        all_seed_normalized.loc[kernel_mask, col_name] = (
                            (all_seed_normalized.loc[kernel_mask, col_name] - min_val)
                            / (max_val - min_val)
                        ).astype(int)

        generalist_normalized_df.append(all_seed_normalized)

    # Now add kernel-specific samples to the generalist dataframes
    for k, exp in experiments.items():
        scaled_df = exp["scaled_space"].copy()
        scaled_df["kernel"] = k
        generalist_scaled_df.append(scaled_df)

        normalized_df = exp["normalized_space"].copy()
        normalized_df["kernel"] = k
        generalist_normalized_df.append(normalized_df)

    # Combine all dataframes
    generalist_scaled = pd.concat(generalist_scaled_df, ignore_index=True)
    generalist_normalized = pd.concat(generalist_normalized_df, ignore_index=True)

    # Save generalist dataframes
    generalist_scaled_file = "generalist_scaled_samples.csv"
    generalist_normalized_file = "generalist_normalized_samples.csv"

    generalist_scaled.to_csv(generalist_scaled_file, index=False)
    generalist_normalized.to_csv(generalist_normalized_file, index=False)

    output_files.extend([generalist_scaled_file, generalist_normalized_file])

    # Add generalist dataframes to experiments dictionary
    experiments["generalist"] = {
        "scaled_space": generalist_scaled,
        "normalized_space": generalist_normalized
    }

    return experiments, output_files


def build_gpu_experiments(
    gpu_space: dict[str, t.Any], experimental_space: pd.DataFrame
) -> pd.DataFrame:
    for k, v in gpu_space.items():
        experimental_space[k] = v

    return experimental_space


def run_experiments(
    kernel_functions: dict[str, t.Any],
    kernel_experiments: dict[str, t.Any],
    kernel_iterations: int,
    use_mean: bool
) -> list[str]:
    output_files = []

    for kernel, k_call in kernel_functions.items():
        measurements = np.array([])

        scaled_space = kernel_experiments[kernel]["scaled_space"]
        normalized_space = kernel_experiments[kernel]["normalized_space"]

        for index, row in scaled_space.iterrows():
            print(f"{kernel}: iteration {index}")
            new_measurements = None
            try:
                if kernel == "attn":
                    seq_len_param = "max_seq_length" if "max_seq_length" in row.index else "seq_length"
                    new_measurements = k_call(
                        int(row["batch"]),
                        int(row[seq_len_param]),
                        int(row[seq_len_param]),
                        int(row["n_embed"]),
                        int(row["n_head"]),
                        int(row["tensor_parallelism"]),
                        int(row["phase"]),
                        num_iters=kernel_iterations,
                    )
                elif kernel == "mlp":
                    new_measurements = k_call(
                        int(row["batch"]),
                        int(row["seq_length"]),
                        int(row["n_embed"]),
                        int(row["n_head"]),
                        int(row["tensor_parallelism"]),
                        int(row["phase"]),
                        num_iters=kernel_iterations,
                    )
                else:
                    new_measurements = k_call(
                        int(row["batch"]),
                        int(row["seq_length"]),
                        int(row["n_embed"]),
                        int(row["phase"]),
                        num_iters=kernel_iterations,
                    )
                measurements = np.append(
                    measurements,
                    new_measurements.mean() if use_mean else new_measurements,
                )
                print(f"{kernel}: Row {index} succeeded!")

            except Exception as e:
                print(f"{kernel}: Row {index} failed with error: {str(e)}")
                print(f"{kernel}: parameters\n{row}")
                measurements = np.append(measurements,
                                       np.repeat(
                                           pd.NA,
                                           kernel_iterations if not use_mean else 1,
                                       ))

        scaled_experiments = pd.DataFrame(
            scaled_space.values.repeat(
                kernel_iterations if not use_mean else 1,
                axis=0
            ), columns=scaled_space.columns
        )

        normalized_experiments = pd.DataFrame(
            normalized_space.values.repeat(
                kernel_iterations if not use_mean else 1,
                axis=0
            ),
            columns=normalized_space.columns,
        )

        scaled_experiments["runtime_s"] = measurements
        scaled_experiments["kernel"] = kernel

        normalized_experiments["runtime_s"] = measurements
        normalized_experiments["kernel"] = kernel

        scaled_output = f"{kernel}_scaled_experiments.csv"
        normalized_output = f"{kernel}_normalized_experiments.csv"

        scaled_experiments.to_csv(scaled_output, index=False)
        normalized_experiments.to_csv(normalized_output, index=False)

        output_files.extend([scaled_output, normalized_output])

    return output_files


def simulate_experiment_runs(kernel_functions, kernel_experiments, kernel_iterations):
    """Simulate experiment runs without actually executing them."""
    planned_runs = []

    for kernel, k_call in kernel_functions.items():
        scaled_space = kernel_experiments[kernel]["scaled_space"]

        for index, row in scaled_space.iterrows():
            if kernel == "attn":
                seq_len_param = "max_seq_length" if "max_seq_length" in row.index else "seq_length"
                planned_runs.append(
                    f"{kernel}(batch={int(row['batch'])}, "
                    f"seq_len={int(row[seq_len_param])}, "
                    f"max_seq_len={int(row[seq_len_param])}, "
                    f"n_embed={int(row['n_embed'])}, "
                    f"num_heads={int(row['n_head'])}, "
                    f"tensor_parallelism={int(row['tensor_parallelism'])}, "
                    f"phase={int(row['phase'])}, "
                    f"num_iters={kernel_iterations})"
                )
            elif kernel == "mlp":
                planned_runs.append(
                    f"{kernel}(batch={int(row['batch'])}, "
                    f"seq_len={int(row['seq_length'])}, "
                    f"n_embed={int(row['n_embed'])}, "
                    f"num_heads={int(row['n_head'])}, "
                    f"tensor_parallelism={int(row['tensor_parallelism'])}, "
                    f"phase={int(row['phase'])}, "
                    f"num_iters={kernel_iterations})"
                )
            else:
                planned_runs.append(
                    f"{kernel}(batch={int(row['batch'])}, "
                    f"seq_len={int(row['seq_length'])}, "
                    f"n_embed={int(row['n_embed'])}, "
                    f"phase={int(row['phase'])}, "
                    f"num_iters={kernel_iterations})"
                )

    return planned_runs

def check_sample_files_exist(kernel_search_spaces):
    """Check if all necessary sample files exist."""
    required_files = []

    # Individual kernel seed sample files
    for kernel in kernel_search_spaces.keys():
        required_files.append(f"{kernel}_seed_sample.csv")

    # Combined seed sample file
    required_files.append("all_kernels_seed_sample.csv")

    # Generalist sample files
    required_files.append("generalist_scaled_samples.csv")
    required_files.append("generalist_normalized_samples.csv")

    # Check if files exist
    missing_files = [file for file in required_files if not os.path.exists(file)]

    return len(missing_files) == 0, missing_files

def profiler_main(
    mode: str,
    sample_exponent: int,
    device: str,
    phase: int,
    gpu_name: str,
    kernel_iterations: int,
    use_mean: bool,
    include_sobol: bool
):
    """Main profiler function that handles different modes of operation."""

    kernel_functions = {
        "attn": benchmark_llama_attn,
        "mlp": benchmark_llama_mlp,
        "res_add": benchmark_llm_residual_addition,
        "rms_norm": benchmark_llm_rmsnorm,
    }

    gpu_search_space = pd.DataFrame(
        {
            "gpu_name": ["h100", "a100", "v100"],
            "sms": [114, 108, 80],
            "tpcs": [57, 54, 40],
            "gpcs": [8, 7, 6],
            "cuda_cores": [14592, 6912, 5120],
            "tensor_cores": [456, 432, 640],
            "ram_gb": [80, 40, 32],
            "l2_cache_mb": [50, 40, 6],
            "mem_bw_gB_s": [2000, 1555, 900],
        }
    )

    kernel_search_spaces = {
        "attn": {
            "batch": [2**i for i in range(0, 6)],
            "max_seq_length": [2**i for i in range(10, 13)],
            "n_embed": [2**i for i in range(11, 14)],
            "n_head": [2**i for i in range(4, 7)],
            "tensor_parallelism": [2**i for i in range(0, 2)],
        },
        "mlp": {
            "batch": [2**i for i in range(0, 6)],
            "seq_length": [2**i for i in range(10, 13)],
            "n_embed": [2**i for i in range(11, 14)],
            "n_head": [2**i for i in range(4, 7)],
            "tensor_parallelism": [2**i for i in range(0, 2)],
        },
        "res_add": {
            "batch": [2**i for i in range(0, 6)],
            "seq_length": [2**i for i in range(10, 13)],
            "n_embed": [2**i for i in range(11, 14)],
        },
        "rms_norm": {
            "batch": [2**i for i in range(0, 6)],
            "seq_length": [2**i for i in range(10, 13)],
            "n_embed": [2**i for i in range(11, 14)],
        },
    }

    # Expanded search space for Sobol samples (quasi-random design)
    sobol_kernel_search_spaces = {
        "attn": {
            "batch": [2**i for i in range(0, 8)],  # Expanded to 128
            "max_seq_length": [2**i for i in range(8, 15)],  # 256 to 16384
            "n_embed": [2**i for i in range(10, 15)],  # 1024 to 16384
            "n_head": [2**i for i in range(3, 8)],  # 8 to 128
            "tensor_parallelism": [2**i for i in range(0, 4)],  # 1 to 8
        },
        "mlp": {
            "batch": [2**i for i in range(0, 8)],
            "seq_length": [2**i for i in range(8, 15)],
            "n_embed": [2**i for i in range(10, 15)],
            "n_head": [2**i for i in range(3, 8)],
            "tensor_parallelism": [2**i for i in range(0, 4)],
        },
        "res_add": {
            "batch": [2**i for i in range(0, 8)],
            "seq_length": [2**i for i in range(8, 15)],
            "n_embed": [2**i for i in range(10, 15)],
        },
        "rms_norm": {
            "batch": [2**i for i in range(0, 8)],
            "seq_length": [2**i for i in range(8, 15)],
            "n_embed": [2**i for i in range(10, 15)],
        },
    }

    # Define value ranges for Sobol sampling using the expanded search space
    sobol_value_ranges = {
        k: {p: [v[0], v[-1]] for p, v in sobol_kernel_search_spaces[k].items()}
        for k in sobol_kernel_search_spaces.keys()
    }

    defaults = {
        "phase": phase,
    }

    if mode == 'dry-run':
        # Generate seed samples and check what files would be created
        _, _, seed_output_files = generate_seed_samples(kernel_search_spaces)

        # Generate samples with Sobol only if include_sobol is True
        samples = {}
        if include_sobol:
            samples = sobol_sample(sobol_value_ranges, sobol_kernel_search_spaces, sample_exponent=sample_exponent)
        else:
            # Create empty samples structure when Sobol is excluded
            samples = {k: {
                "normalized_sample": np.empty((0, len(sobol_kernel_search_spaces[k]))),
                "scaled_sample": np.empty((0, len(sobol_kernel_search_spaces[k])))
            } for k in sobol_kernel_search_spaces.keys()}

        # Build experiments (but don't write files)
        selected_gpu = gpu_search_space[gpu_search_space["gpu_name"] == gpu_name]
        if selected_gpu.empty:
            print(f"Error: GPU '{gpu_name}' not found. Available options are: {', '.join(gpu_search_space['gpu_name'])}")
            return 1

        # Pass empty dictionaries for seed samples in dry-run mode
        experiments, exp_output_files = build_experiments(
            kernel_search_spaces,
            sobol_kernel_search_spaces,
            samples,
            defaults,
            selected_gpu,
            {}, # Empty seed_samples for dry run
            pd.DataFrame(), # Empty all_seed_df
        )

        # List experiment output files
        experiment_output_files = []
        for kernel in kernel_functions.keys():
            experiment_output_files.append(f"{kernel}_scaled_experiments.csv")
            experiment_output_files.append(f"{kernel}_normalized_experiments.csv")

        # Print all files that would be created
        print("The following files would be created:")
        for file in seed_output_files + exp_output_files + experiment_output_files:
            print(f"  - {file}")

        return 0

    elif mode == 'prepare-samples':
        # Generate seed samples
        seed_samples, all_seed_df, seed_output_files = generate_seed_samples(kernel_search_spaces)

        # Generate samples with Sobol only if include_sobol is True
        samples = {}
        if include_sobol:
            samples = sobol_sample(sobol_value_ranges, sobol_kernel_search_spaces, sample_exponent=sample_exponent)
        else:
            # Create empty samples structure when Sobol is excluded
            samples = {k: {
                "normalized_sample": np.empty((0, len(sobol_kernel_search_spaces[k]))),
                "scaled_sample": np.empty((0, len(sobol_kernel_search_spaces[k])))
            } for k in sobol_kernel_search_spaces.keys()}

        # Build experiments
        selected_gpu = gpu_search_space[gpu_search_space["gpu_name"] == gpu_name]
        if selected_gpu.empty:
            print(f"Error: GPU '{gpu_name}' not found. Available options are: {', '.join(gpu_search_space['gpu_name'])}")
            return 1

        experiments, exp_output_files = build_experiments(
            kernel_search_spaces,
            sobol_kernel_search_spaces,
            samples,
            defaults,
            selected_gpu,
            seed_samples,
            all_seed_df,
        )

        # Filter out generalist from experiments when simulating runs
        kernel_experiments = {k: v for k, v in experiments.items() if k in kernel_functions}

        # Save the experiment dataframes without running benchmarks
        experiment_output_files = []
        for kernel, exp_data in kernel_experiments.items():
            # Create dataframes with placeholder runtimes
            scaled_df = exp_data["scaled_space"].copy()
            scaled_df["runtime_s"] = pd.NA  # Placeholder for runtime
            scaled_df["kernel"] = kernel

            normalized_df = exp_data["normalized_space"].copy()
            normalized_df["runtime_s"] = pd.NA  # Placeholder for runtime
            normalized_df["kernel"] = kernel

            # Save to files
            scaled_output = f"{kernel}_scaled_experiments_planned.csv"
            normalized_output = f"{kernel}_normalized_experiments_planned.csv"

            scaled_df.to_csv(scaled_output, index=False)
            normalized_df.to_csv(normalized_output, index=False)

            experiment_output_files.extend([scaled_output, normalized_output])

        # Print what experiments would be run
        planned_runs = simulate_experiment_runs(kernel_functions, kernel_experiments, kernel_iterations)

        print(f"Created {len(seed_output_files + exp_output_files + experiment_output_files)} sample and experiment files.")
        print(f"Would run {len(planned_runs)} benchmark iterations.")
        print("\nSample of planned benchmark runs:")
        for i, run in enumerate(planned_runs[:10]):  # Show first 10 runs
            print(f"  {i+1}. {run}")
        if len(planned_runs) > 10:
            print(f"  ... and {len(planned_runs) - 10} more")

        print("\nSaved experiment files with placeholder runtimes:")
        for file in experiment_output_files:
            print(f"  - {file}")

        return 0

    elif mode == 'run':
        # Generate seed samples
        seed_samples, all_seed_df, _ = generate_seed_samples(kernel_search_spaces)

        # Generate samples with Sobol only if include_sobol is True
        samples = {}
        if include_sobol:
            samples = sobol_sample(sobol_value_ranges, sobol_kernel_search_spaces, sample_exponent=sample_exponent)
        else:
            # Create empty samples structure when Sobol is excluded
            samples = {k: {
                "normalized_sample": np.empty((0, len(sobol_kernel_search_spaces[k]))),
                "scaled_sample": np.empty((0, len(sobol_kernel_search_spaces[k])))
            } for k in sobol_kernel_search_spaces.keys()}

        # Build experiments
        selected_gpu = gpu_search_space[gpu_search_space["gpu_name"] == gpu_name]
        if selected_gpu.empty:
            print(f"Error: GPU '{gpu_name}' not found. Available options are: {', '.join(gpu_search_space['gpu_name'])}")
            return 1

        experiments, _ = build_experiments(
            kernel_search_spaces,
            sobol_kernel_search_spaces,
            samples,
            defaults,
            selected_gpu,
            seed_samples,
            all_seed_df,
        )

        # Filter out generalist from experiments when running
        kernel_experiments = {k: v for k, v in experiments.items() if k in kernel_functions}

        # Run experiments
        output_files = run_experiments(kernel_functions, kernel_experiments, kernel_iterations, use_mean)

        print(f"Successfully created {len(output_files)} output files.")
        return 0

    else:
        print(f"Error: Unknown mode '{mode}'")
        return 1


@click.command()
@click.option('--mode', type=click.Choice(['dry-run', 'prepare-samples', 'run']),
              default='run', help='Operation mode: dry-run checks files, prepare-samples generates samples, run executes benchmarks')
@click.option('--sample-exponent', default=11, type=int, help='Exponent for Sobol sample size (2^n samples)')
@click.option('--device', default=DEFAULT_DEVICE, help='Compute device (cuda or cpu)')
@click.option('--phase', default=0, type=int,
              help='Phase: 0=TRAINING_FWD, 1=TRAINING_BCWD, 2=INFERENCE_TOKENGEN, 3=INFERENCE_PREFILL')
@click.option('--gpu-name', default='a100', help='GPU model name (h100, a100, v100)')
@click.option('--kernel-iterations', default=20, type=int, help='Number of iterations for each kernel run')
@click.option('--use-mean/--no-use-mean', default=True, help='Whether to use mean of iterations')
@click.option('--include-sobol/--exclude-sobol', default=True,
              help='Whether to include Sobol samples in the experiment design')
def cli(mode, sample_exponent, device, phase, gpu_name, kernel_iterations, use_mean, include_sobol):
    """Profile LLM kernels with various configurations."""
    return profiler_main(
        mode=mode,
        sample_exponent=sample_exponent,
        device=device,
        phase=phase,
        gpu_name=gpu_name,
        kernel_iterations=kernel_iterations,
        use_mean=use_mean,
        include_sobol=include_sobol
    )


if __name__ == "__main__":
    cli()
