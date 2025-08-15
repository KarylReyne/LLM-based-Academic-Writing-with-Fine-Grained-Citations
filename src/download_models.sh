mkdir -p reasonir_8b
cd reasonir_8b
huggingface-cli download reasonir/ReasonIR-8B --local-dir . --repo-type model


mkdir -p ../qwen2.5_7b_instruct
cd ../qwen2.5_7b_instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir . --repo-type model


mkdir -p ../deepseek-r1-distill-qwen-7b
cd ../deepseek-r1-distill-qwen-7b
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir . --repo-type model