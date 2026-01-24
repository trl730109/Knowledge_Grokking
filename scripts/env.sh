#!/bin/bash
# Knowledge Grokking 环境配置文件

server=${server:-SZA6000}        # 默认服务器，可以按需改
dnn=${dnn:-qwen2.5-7b-instruct}
model_dir=${model_dir:-None}
cuda_devices=${cuda_devices:-0,1,2,3}

case "$server" in
    "SZA6000")
        case "$dnn" in
            "llama2-7b-chat")       model_dir="/workspace/tzc/models/Llama-2-7b-chat-hf" ;;
            "llama3-8b-Instruct")   model_dir="/workspace/tzc/models/Llama-3-8B-Instruct" ;;
            "mistral-7B-Instruct")  model_dir="/workspace/tzc/models/Mistral-7B-Instruct-v0.3" ;;
            "qwen2.5-7b-instruct")  model_dir="/workspace/tzc/Qwen/Qwen2.5-7B-Instruct" ;;
            "qwen2.5-14b-instruct") model_dir="/workspace/tzc/Qwen/Qwen2.5-14B-Instruct" ;;
            "llama2-13b")           model_dir="/workspace/tzc/models/Llama-2-13b-hf" ;;
            "deepseek-r1-14b")      model_dir="/workspace/tzc/models/DeepSeek-R1-Distill-Qwen-14B" ;;
            *)
                echo "Unknown dnn: $dnn"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unknown cluster name: $server"
        exit 1
        ;;
esac

# 统一导出，方便后续脚本和子进程使用
export server dnn model_dir cuda_devices

