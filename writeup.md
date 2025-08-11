# benchmarking_script

(a) See: [benchmark_model.py](cs336_systems/benchmark_model.py)

(b)
| Model  | Forward Time (s) | Backward Time (s) | Total Time (s) |
| ------ | ---------------- | ----------------- | -------------- |
| small  | 0.035 ± 0.001    | 0.043 ± 0.001     | 0.077 ± 0.001  |
| medium | 0.071 ± 0.001    | 0.082 ± 0.003     | 0.153 ± 0.004  |
| large  | 0.119 ± 0.004    | 0.129 ± 0.005     | 0.247 ± 0.008  |
| xl     | 0.171 ± 0.004    | 0.174 ± 0.003     | 0.345 ± 0.006  |
| 2.7B   | 0.147 ± 0.002    | 0.218 ± 0.000     | 0.365 ± 0.002  |

(c)

# nsys_profile

# benchmarking_mixed_precision
(a)
| Component              | Dtype(s)      |
| ---------------------- | ------------- |
| Model Parameters       | torch.float32 |
| ToyModel.fc1 Output    | torch.float16 |
| ToyModel.ln Output     | torch.float32 |
| Predicted Logits       | torch.float16 |
| Loss                   | torch.float32 |
| Gradients (Parameters) | torch.float32 |

(b) The most sensitive part of layer normalization to mixed precision is probably calculating the variance and dividing by it. When calculating the variance, we take the square of potentially small numbers (the diff from the mean) so this can underflow in fp16. Then, when dividing by the variance, we divide by a potentially small number, so this can overflow in fp16.

(c)