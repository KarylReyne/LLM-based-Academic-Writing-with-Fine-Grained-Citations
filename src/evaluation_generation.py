from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from passage_retrieval_interface import get_config, get_passage_retrieval_models
from scholarcopilot_model import load_model
from dataset_loaders import load_generation_eval_dataset, arxiv_to_corpus_id, scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset


def load_generation_evaluation_model(config):
    judge_config = AutoConfig.from_pretrained(config["judge"])
    judge = AutoModelForCausalLM.from_pretrained(config["judge"], config=judge_config)
    judge.to(config["judge_device"])

    judge_tokenizer = AutoTokenizer.from_pretrained(config["judge"])
    judge_tokenizer.add_tokens(config["special_tokens"])
    # judge.resize_token_embeddings(len(judge_tokenizer))
    print("judge loaded.")
    return judge, judge_tokenizer


if __name__ == "__main__":
    config = get_config()

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)

    judge, judge_tokenizer = load_generation_evaluation_model(config)

    docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    docs_corpus_id_map = arxiv_to_corpus_id(docs_id_map_path, processed_corpus_path)
    docs_arxiv_id_map = {v: k for k, v in docs_corpus_id_map.items()}

    sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
    sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)
    sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
    
    docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

    eval_dataset_path = "data/eval_dataset_generation.jsonl"
    eval_dataset, eval_indices = load_generation_eval_dataset(eval_dataset_path, docs_retrieval_dataset, sc_arxiv_id_map, max_samples=1000, shuffle=True)

    # print(eval_dataset[:3])

    count = 0
    print()
    for i in eval_indices:
        item = eval_dataset[i]
        context = item["context"]

        # generate with SC


        # generate with SC+PR


        # judge each generated output individually (prompt from SC paper)


        # (judge both at once by contrasting them?)


