from collections import defaultdict
from contextlib import contextmanager, nullcontext

import fire
import numpy as np
import torch
import torch.nn.functional as F
from cs336_basics import model
from pydantic import BaseModel

from cs336_systems.annotated_model import (
    annotated_lm_forward,
    annotated_scaled_dot_product_attention,
    annotated_transformer_block_forward,
)
from cs336_systems.profiling_utils import profiling_range


def _patch_model():
    """Patch the model to use the annotated functions."""
    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model.BasicsTransformerLM.forward = annotated_lm_forward
    model.TransformerBlock.forward = annotated_transformer_block_forward


class TransformerLMConfig(BaseModel):
    vocab_size: int = 10_000
    context_length: int | None = None
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0


LM_CONFIGS = {
    "small": TransformerLMConfig(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ),
    "medium": TransformerLMConfig(
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
    ),
    "large": TransformerLMConfig(
        d_model=1280,
        d_ff=5120,
        num_layers=36,
        num_heads=20,
    ),
    "xl": TransformerLMConfig(
        d_model=1600,
        d_ff=6400,
        num_layers=48,
        num_heads=25,
    ),
    "2.7B": TransformerLMConfig(
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
    ),
}


def get_model_from_config(config: TransformerLMConfig):
    return model.BasicsTransformerLM(**config.model_dump())


def benchmark_model(
    model_config: TransformerLMConfig,
    sequence_length: int,
    warmup_steps: int = 3,
    measurement_steps: int = 5,
    batch_size: int = 4,
    device: str = "cpu",
    use_amp: bool = False,
):
    model_config.context_length = sequence_length
    model = get_model_from_config(model_config).to(device)

    batch_inputs = torch.randint(0, model_config.vocab_size, (batch_size, sequence_length)).to(device)
    batch_targets = torch.randint(0, model_config.vocab_size, (batch_size, sequence_length)).to(device)

    amp_context = torch.autocast(device_type=device, dtype=torch.float16) if use_amp else nullcontext()

    def step():
        with amp_context:
            with profiling_range("forward") as forward_range:
                outputs = model(batch_inputs)
            forward_time = forward_range.elapsed

            loss = F.cross_entropy(outputs.reshape(-1, model_config.vocab_size), batch_targets.reshape(-1))

            with profiling_range("backward") as backward_range:
                loss.backward()
            backward_time = backward_range.elapsed

            return {
                "forward_time": forward_time,
                "backward_time": backward_time,
                "total_time": forward_time + backward_time,
            }

    with profiling_range("warmup"):
        for _ in range(warmup_steps):
            step()

    measurements = defaultdict(list)
    with profiling_range("measurement"):
        for _ in range(measurement_steps):
            with profiling_range("step"):
                step_measurements = step()
            for key, value in step_measurements.items():
                measurements[key].append(value)

    return measurements


def print_measurements(measurements: dict[str, list[float]]):
    for key, values in measurements.items():
        print(f"{key}: {np.mean(values):.3f} ± {np.std(values):.3f} seconds")


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@contextmanager
def _profile_memory(output_filename: str):
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    yield
    torch.cuda.memory._dump_snapshot(f"{output_filename}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


def main(
    model_name: str,
    sequence_length: int = 128,
    batch_size: int = 4,
    warmup_steps: int = 5,
    measurement_steps: int = 10,
    device: str = get_best_device(),
    patch_model: bool = True,
    use_amp: bool = False,
    profile_memory: bool = False,
):
    profile_memory_context = _profile_memory(f"tmp/memory_snapshot_{model_name}") if profile_memory else nullcontext()

    if patch_model:
        _patch_model()

    with profile_memory_context:
        measurements = benchmark_model(
            model_config=LM_CONFIGS[model_name],
            sequence_length=sequence_length,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            device=device,
            use_amp=use_amp,
        )
    print(f"{model_name}:")
    print_measurements(measurements)


if __name__ == "__main__":
    fire.Fire(main)
