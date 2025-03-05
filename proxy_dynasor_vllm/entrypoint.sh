#!/bin/bash
# Nếu biến MT chưa được set, gán giá trị mặc định
: "${MT:=http://vllm-serve-deepSeek-r1-qwen14b:8000}"

# Lấy model từ API (sử dụng endpoint /v1/models)
echo "Fetching models from ${MT}/v1/models ..."
MODEL=$(curl -s "${MT}/v1/models" | jq -r '.data[0].id')
if [ -z "$MODEL" ] || [ "$MODEL" == "null" ]; then
    echo "No model found from ${MT}/v1/models, using default model name."
    MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
fi
echo "Using model: $MODEL"

# Khởi chạy Dynasor với monkey patch đã được import
exec python start_dynasor.py --base-url "$MT" --model "$MODEL"
