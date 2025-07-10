import json
import sys
import itertools

from passage_retrieval_interface import get_config, get_passage_retrieval_models
from util import recall_at_k


def evaluate_scholarcopilot(dataset_path):
    pass


def evaluate_scholarcopilot_with_passageretrieval(dataset_path):
    pass


if __name__ == "__main__":
    RECALL_K = 1

    config = get_config()
    passage_retrieval_models = get_passage_retrieval_models(config)

    