cd scholarcopilot_train/src

EMBEDDING_OUTPUT_DIR="../../data/"
dataset_path="../../data/documents_3.0_processed_corpus.jsonl"
model_path="../../scholarcopilot_model_v1208"

for s in 0
do
echo ${s}
gpuid=$s
CUDA_VISIBLE_DEVICES=$gpuid python -m encode \
  --output_dir temp \
  --model_name_or_path ${model_path} \
  --bf16 \
  --pooling eos \
  --normalize \
  --per_device_eval_batch_size 16 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_name json \
  --dataset_path ${dataset_path} \
  --dataset_number_of_shards 1 \
  --dataset_shard_index ${s} \
  --encode_output_path ${EMBEDDING_OUTPUT_DIR}/corpus.${s}.pkl
done