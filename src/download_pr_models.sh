mkdir -p reasonir_8b
cd reasonir_8b

huggingface-cli download reasonir/ReasonIR-8B --local-dir . --repo-type model


mkdir -p ../qwen2.5_7b_instruct
cd ../qwen2.5_7b_instruct

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir . --repo-type model
