import torch
import json
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch import nn
import numpy as np
import itertools
from datetime import datetime

from latex_parsing import download_from_arxiv, search_arxiv_for_citations_data, get_source_citations, LABEL_SEPARATOR, CITATION_MASK
from retrieval import retrieval
from reranking import reranking_and_scoring


def save_results(
    evaluation_records,
    id,
    target_citation_record,
    retrieval_instruction_query,
    retrieval_instruction_document,
    reranking_instruction,
    config
):
    results = None
    try:
        with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/evaluation_records.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        if not os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}'):
            os.makedirs(f'out/{datetime.now().strftime('%Y-%m-%d')}')
        results = {}
    assert os.path.exists(f'out/{datetime.now().strftime('%Y-%m-%d')}')
    assert results != None

    results[f"{datetime.now().strftime('%H-%M-%S')}"] = {
        "query_doc_id": f"{id}",
        "target_doc_id": f"{target_citation_record["arxiv_id"]}",
        "config": config,
        "instructions": {
            "retr_query": retrieval_instruction_query,
            "retr_document": retrieval_instruction_document,
            "rera_scoring": reranking_instruction("", "")
        },
        "records": evaluation_records
    }

    with open(f'out/{datetime.now().strftime('%Y-%m-%d')}/evaluation_records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# srun --job-name "FineGrainedCitations" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    config = None
    with open('cfg/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None

    # retrieval model definition
    retr_tokenizer = AutoTokenizer.from_pretrained(config["retriever"])
    retriever = AutoModel.from_pretrained(
        config["retriever"], 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    retriever = retriever.to("cuda")
    retriever.eval()

    # reranker model definition
    rera_tokenizer = AutoTokenizer.from_pretrained(config["reranker"])
    reranker = AutoModelForCausalLM.from_pretrained(
        config["reranker"], 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    reranker = reranker.to("cuda")

    id = "2108.09084" # Fastformer
    citations_data, _ = search_arxiv_for_citations_data(id, additional_citation_records={
        "vaswani2017attention": {
            "bib_id": "vaswani2017attention",
            "arxiv_id": "1706.03762", 
            "title": "Attention is all you need", 
            "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.", 
            "name": "Ashish Vaswani Noam Shazeer Niki Parmar Jakob Uszkoreit Llion Jones Aidan~N Gomez Lukasz Kaiser Illia Polosukhin"
        }
    })
    target_bib_id = "vaswani2017attention" # Transformer
    target_citation_record = citations_data[target_bib_id]
    (citing_sents, citing_context), target_doc_sections = get_source_citations(
        id, 
        target_citation_record, 
        retr_tokenizer, 
        with_chunking=config["target_chunking"], 
        chunk_size=config["chunk_size"],
        query_expansion_method=config["query_expansion_method"],
        query_context_size=config["query_context"]
    )


    # instructions (based on ReasonIR / BRIGHT)
    retrieval_instruction_query = f"<|user|>\nGiven a query with a citation marked by '{CITATION_MASK}', retrieve relevant passages that describe the cited topic\n<|embed|>\n"
    retrieval_instruction_document = f"<|embed|>\n"
    # retrieval_instruction_query = f""
    # retrieval_instruction_document = f""

    reranking_instruction = lambda q, d: f"You are given a query with a citation marked by '{CITATION_MASK}' and a list of paragraphs. A paragraph is relevant if it describes or contains information about the cited topic. A paragraph is not relevant if it doesn't contain information about the cited topic, even if it mentions similar topics. Rank all paragraphs below based on how relevant to the query they are. Following the order of the passages below, your answer should be 'Relevance scores: X' where X is a list of numbers from 0-10 where each number is the score of the corresponding paragraph. 0 means completely irrelevant, 10 means highly relevant and completely addresses the query. Don't output anything else. Here is the query: <start_query>{q}<end_query>Here are the paragraphs: {"".join([f"<start_paragraph-{i+1}>{p}<end_paragraph-{i+1}>" for i, p in enumerate(d)])} <start_example_answer>Relevance scores: [<score for paragraph-1>, <score for paragraph-2>, ...]<end_example_answer>"


    evaluation_records = {}
    # queries = citing_sents if not config["expand_query"] else citing_context
    queries = citing_context
    num_queries = len(queries)

    evaluation_records = {}
    for i, q in enumerate(queries):
        evaluation_records[f"query-{i}"] = {
            f"query-{i}": q,
            "retrieved documents": {},
            "reranked documents": {},
            "final ranking": {}
        }


    # --- RETRIEVAL ---
    # prepare data for batch processing

    # separate section labels and documents
    doc_lbls = []
    docs = []
    for d in target_doc_sections:
        split = d.split(LABEL_SEPARATOR)
        doc_lbls.append(split[0])
        docs.append(split[1])

    queries = [list(itertools.repeat(q, len(target_doc_sections))) for q in queries] # num_queries x num_sections
    document_labels = list(itertools.repeat(doc_lbls, num_queries)) # num_queries x num_sections
    documents = list(itertools.repeat(docs, num_queries)) # num_queries x num_sections

    assert len(queries) == len(document_labels) == len(documents), f"{len(queries)}, {len(document_labels)}, {len(documents)}"
    assert len(queries[0]) == len(document_labels[0]) == len(documents[0]), f"{len(queries[0])}, {len(document_labels[0])}, {len(documents[0])}"

    retrieval(
        evaluation_records, 
        num_queries, 
        queries, 
        document_labels, 
        documents, 
        retrieval_instruction_query, 
        retrieval_instruction_document, 
        retriever, 
        config
    )


    # --- RERANKING ---
    # reranking_inputs = [] # num_queries x topk_retrieval
    # document_labels = [] # num_queries x topk_retrieval
    # documents = [] # num_queries x topk_retrieval

    # for i in range(num_queries):
    #     qry_reranking_inputs = []
    #     qry_document_labels = []
    #     qry_documents = []
    #     for doc_label, doc_dict in evaluation_records[f"query-{i}"]["retrieved documents"].items():
    #         q = evaluation_records[f"query-{i}"][f"query-{i}"]
    #         d = doc_dict["section chunk"]
    #         qry_reranking_inputs.append(rera_tokenizer.apply_chat_template([
    #                 {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
    #                 {"role": "user", "content": reranking_instruction(q, d)}
    #             ],
    #             tokenize=False,
    #             add_generation_prompt=True
    #         ))
    #         qry_document_labels.append(doc_label)
    #         qry_documents.append(d)
    #     reranking_inputs.append(qry_reranking_inputs)
    #     document_labels.append(qry_document_labels)
    #     documents.append(qry_documents)

    # assert len(reranking_inputs) == num_queries, f"{len(reranking_inputs)}, {num_queries}"
    # assert len(reranking_inputs[0]) == config["retriever_topk"], f"{len(reranking_inputs[0])}, {config["retriever_topk"]}"

    single_queries = [] # num_queries x 1 (not repeated like queries)
    document_labels = [] # num_queries x topk_retrieval
    documents = [] # num_queries x topk_retrieval

    for i in range(num_queries):
        qry_document_labels = []
        qry_documents = []
        for doc_label, doc_dict in evaluation_records[f"query-{i}"]["retrieved documents"].items():
            qry_document_labels.append(doc_label)
            qry_documents.append(doc_dict["section chunk"])
        single_queries.append(evaluation_records[f"query-{i}"][f"query-{i}"])
        document_labels.append(qry_document_labels)
        documents.append(qry_documents)

    assert len(single_queries) == num_queries, f"{len(single_queries)}, {num_queries}"
    assert len(documents[0]) == config["retriever_topk"], f"{len(documents[0])}, {config["retriever_topk"]}"

    reranking_and_scoring(
        evaluation_records, 
        num_queries, 
        single_queries, 
        document_labels, 
        documents, 
        reranking_instruction, 
        reranker, 
        rera_tokenizer, 
        config
    )

    save_results(
        evaluation_records, 
        id, 
        target_citation_record, 
        retrieval_instruction_query, 
        retrieval_instruction_document, 
        reranking_instruction, 
        config
    )
    