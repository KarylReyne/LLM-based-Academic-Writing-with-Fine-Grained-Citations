

from passage_retrieval_interface import get_config, get_passage_retrieval_models, retrieve_relevant_passages, apply_retrieval_context_window
from scholarcopilot_model import load_model
from dataset_loaders import load_pr_train_set_for_scholarcopilot



if __name__ == "__main__":
    config = get_config("cfg/config_training_dataset_construction.json")

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)
    
    pr_train_dataset_path = "data/training_dataset_-_sc_for_passage_retrieval.jsonl"
    docs_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_fulltext_dataset_path = "data/documents_3.0_with_ids_only_fulltext.jsonl"
    get_passage_from_context = lambda ctx, refs: retrieve_relevant_passages(
        apply_retrieval_context_window(ctx, tokenizer, config), 
        refs,
        passage_retrieval_models, 
        config, 
        silent=True, 
        save_passage_records=False
    )["ranked_passages"][0]
    load_pr_train_set_for_scholarcopilot(pr_train_dataset_path, docs_fulltext_dataset_path, docs_dataset_path, get_passage_from_context, tokenizer, config)