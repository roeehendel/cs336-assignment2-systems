#!/bin/zsh

for MODEL in small medium large xl 2.7B; do
  echo "Profiling model_name=${MODEL}"
  echo "No AMP"
  python cs336_systems/benchmark_model.py --model_name "${MODEL}" --use_amp False
  echo "AMP"
  python cs336_systems/benchmark_model.py --model_name "${MODEL}" --use_amp True
done