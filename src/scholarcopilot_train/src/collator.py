import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from arguments import DataArguments
import torch


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (paper, passages, passages_idx) triples
        :return: tokenized query_ids, passage_ids
        """

        all_papers = [f[0] for f in features] # num_papers x len_paper
        all_targets = [] # num_papers x 4
        all_targets_idx = [] # num_papers x 4, index of each target in the list of all citations in the paper
        for f in features:
            all_targets.extend(f[1])
            all_targets_idx.append(f[2])
        papers_tokenized = self.tokenizer(
            all_papers,
            padding=True, 
            truncation=True,
            max_length=self.data_args.query_max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        targets_tokenized = self.tokenizer(
            all_targets,
            padding=True,
            truncation=True,
            max_length=self.data_args.passage_max_len,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        
        # get the positions of each <cite_start> token (not only the targets) in the paper
        # there is a different number of <cite_start> tokens in each paper!
        cite_start_positions = [] # num_papers x num_cite_tokens_in_paper
        for i, tokenized_paper in enumerate(papers_tokenized['input_ids']):
            cite_start_positions.append([j for j, x in enumerate(tokenized_paper) if x == self.tokenizer.convert_tokens_to_ids('<|cite_start|>')])
        
        # get the positions of each targets' <cite_start> token in the paper
        targets_cite_start_positions = [] # num_papers x 4
        for i, cite_positions in enumerate(cite_start_positions):
            targets_cite_start_positions.append([cite_positions[j] for j in all_targets_idx[i]])
        
        # prepare the label for the model based on input_ids
        # for token_id between <cite_start> and <cite_end>, the label is -100
        # all other labels are the token_id itself
        # -> 'masks' each citation in the paper
        labels = [] # num_papers x len_paper
        for i, tokenized_paper in enumerate(papers_tokenized['input_ids']):
            label = []
            is_ignore = False
            for j, token_id in enumerate(tokenized_paper):
                if token_id == self.tokenizer.convert_tokens_to_ids('<|cite_start|>'):
                    label.append(token_id)
                    is_ignore = True
                elif token_id == self.tokenizer.convert_tokens_to_ids('<|cite_end|>'):
                    label.append(-100)
                    is_ignore = False
                else:
                    if is_ignore:
                        label.append(-100)
                    else:
                        label.append(token_id)
            labels.append(label)
        
        # convert everything to tensor
        papers_tokenized = {k: torch.tensor(v) for k, v in papers_tokenized.items()}
        targets_tokenized = {k: torch.tensor(v) for k, v in targets_tokenized.items()}
        # labels and targets_cite_start_positions are lists of lists with equal length so we can convert them to tensor directly
        labels = torch.tensor(labels)
        targets_cite_start_positions = torch.tensor(targets_cite_start_positions)
        papers_tokenized['labels'] = labels
        papers_tokenized['targets_cite_start_positions'] = targets_cite_start_positions
        return papers_tokenized, targets_tokenized


@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=True, 
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts