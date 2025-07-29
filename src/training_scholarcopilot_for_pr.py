

from passage_retrieval_interface import get_config, get_passage_retrieval_models
from scholarcopilot_model import load_model
from dataset_loaders import load_pr_train_set_for_scholarcopilot


if __name__ == "__main__":
    # config = get_config()

    # model_path = "scholarcopilot_model_v1208/"
    # model, tokenizer = load_model(model_path, config)

    # passage_retrieval_models = get_passage_retrieval_models(config)
    
    pr_train_dataset_path = "data/training_dataset_sc_for_passage_retrieval.jsonl"
    sc_train_dataset_path = "scholarcopilot_data/scholar_copilot_train_data_500k.json"
    load_pr_train_set_for_scholarcopilot(pr_train_dataset_path, sc_train_dataset_path)