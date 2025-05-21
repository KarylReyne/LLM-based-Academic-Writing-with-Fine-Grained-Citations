import torch
import json
import sys
from transformers import AutoModel, AutoTokenizer
from torch import nn
from datetime import datetime
import statistics as stat

from latex_parsing import download_from_arxiv, search_arxiv_for_citations_data, get_source_citations, LABEL_SEPARATOR


# models
RETRIEVER_MODEL = "reasonir/ReasonIR-8B"
# self-consistency calls
M = 10
# target doc sections chunking
ENABLE_CHUNKING = True
CHUNK_SIZE = 512
# query context
EXPAND_QUERY_CONTEXT = True
QUERY_CONTEXT_SIZE = 128
# retrieval
TOPK_RETR = 10
TOPK_RERA = 5 # reserved for the reranker


# srun --job-name "ReasonIRtest" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":


    # TODOs
    # experiment with instructions, specify the mask token

    # https://arxiv.org/abs/2505.12570

    # maybe add ReasonIR's QwenRerank


    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
    model = AutoModel.from_pretrained(RETRIEVER_MODEL, torch_dtype="auto", trust_remote_code=True)
    model = model.to("cuda")
    model.eval()


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
        tokenizer, 
        with_chunking=ENABLE_CHUNKING, 
        chunk_size=CHUNK_SIZE,
        include_query_context=EXPAND_QUERY_CONTEXT,
        query_context_size=QUERY_CONTEXT_SIZE
    )

    # retrieval instructions
    query_instruction = "" 
    doc_instruction = ""

    evaluation_records = {}
    queries = citing_sents if not EXPAND_QUERY_CONTEXT else citing_context
    for query_idx, query in enumerate(queries):

        query_records = {
            f"query-{query_idx}": query,
            "retrieved_documents": {} # filled later
        }
        retrieved_documents = {}

        print() # for console progress report

        for doc_idx, candidate_chunk in enumerate(target_doc_sections):

            sys.stdout.write("\033[F")
            print(f"processing query {query_idx+1}/{len(citing_sents)} - section {doc_idx+1}/{len(target_doc_sections)}")

            t = candidate_chunk.split(LABEL_SEPARATOR)
            chunk_label = t[0]
            chunk = t[1]

            # score aggregation for simple self-consistency, see https://arxiv.org/abs/2505.12570 p.3 chapter 3
            similarity_scores = []
            for _ in range(M):
                query_emb = model.encode(query, instruction=query_instruction)
                doc_emb = model.encode(chunk, instruction=doc_instruction)
                similarity_scores.append(query_emb @ doc_emb.T)

            sim = stat.mean(similarity_scores)

            retrieved_documents[chunk_label] = {
                "section chunk": chunk,
                "sim": f"{sim}"
            }

        # get topk retrieved documents
        retrieved_documents = dict(sorted(retrieved_documents.items(), key=lambda item: item[1]["sim"], reverse=True)[:TOPK_RETR])

        query_records["retrieved_documents"] = retrieved_documents
        evaluation_records[f"query-{query_idx}"] = query_records

        # DEBUG
        # break


    results = None
    try:
        with open('out/evaluation_records.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    assert results != None

    results[f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"] = {
        "query_doc_id": f"{id}",
        "target_doc_id": f"{target_citation_record["arxiv_id"]}",
        "config": {
            "retriever": RETRIEVER_MODEL,
            "m": M,
            "target_chunking": ENABLE_CHUNKING,
            "chunk_size": CHUNK_SIZE,
            "expand_query": EXPAND_QUERY_CONTEXT,
            "query_context": QUERY_CONTEXT_SIZE,
            "retriever_topk": TOPK_RETR,
            "reranker_topk": TOPK_RERA
        },
        "records": evaluation_records
    }

    with open('out/evaluation_records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    