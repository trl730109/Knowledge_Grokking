#!/bin/bash
# model_path=/share/models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
model_path=/workspace/tzc/Qwen/Qwen2.5-7B-Instruct
# 使用显卡 4, 5, 6, 7 启动 7B 判定模型
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --tensor-parallel-size 4 \
    --port 8001 \
    --trust-remote-code \
    --served-model-name local-judge \
    --max-model-len 3072 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --enforce-eager