#!/bin/zsh

# Profile all model sizes and context lengths using benchmark.sh
# Model sizes: small, medium, large, xl, 2.7B
# Context lengths: 128, 256, 512, 1024

for MODEL in small medium large xl 2.7B; do
  for SEQ_LEN in 128 256 512 1024; do
    echo "Profiling model_name=${MODEL} sequence_length=${SEQ_LEN}"
    nsys profile -o "tmp/result_${MODEL}_${SEQ_LEN}" --pytorch autograd-nvtx --python-backtrace=cuda \
      python cs336_systems/benchmark_model.py --model_name "${MODEL}" --sequence_length "${SEQ_LEN}"
  done
done