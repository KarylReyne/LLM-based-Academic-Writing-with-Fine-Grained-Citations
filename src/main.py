import torch
import json
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch import nn
import numpy as np
import itertools
from datetime import datetime
import statistics as stat

from latex_parsing import download_from_arxiv, search_arxiv_for_citations_data, get_source_citations, LABEL_SEPARATOR, CITATION_MASK


# models
RETRIEVER_MODEL = "reasonir/ReasonIR-8B"
RERANKER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# self-consistency calls
M_RETR = 10
M_RERA = 10
# target doc sections chunking
ENABLE_CHUNKING = True
CHUNK_SIZE = 512
# query context
EXPAND_QUERY_CONTEXT = True
QUERY_CONTEXT_SIZE = 128
# retrieval
TOPK_RETR = 10
TOPK_RERA = 5
BATCH_SIZE = 20
# final scoring
DELTA = 0.5


# srun --job-name "FineGrainedCitations" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    # retrieval model definition
    retr_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
    retriever = AutoModel.from_pretrained(RETRIEVER_MODEL, torch_dtype="auto", trust_remote_code=True)
    retriever = retriever.to("cuda")
    retriever.eval()

    # reranker model definition
    rera_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker = AutoModelForCausalLM.from_pretrained(RERANKER_MODEL, torch_dtype="auto", trust_remote_code=True)
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
        with_chunking=ENABLE_CHUNKING, 
        chunk_size=CHUNK_SIZE,
        include_query_context=EXPAND_QUERY_CONTEXT,
        query_context_size=QUERY_CONTEXT_SIZE
    )


    # retrieval instructions (based on ReasonIR / BRIGHT)
    retrieval_instruction_query = f"<|user|>\nGiven a query with a citation marked by '{CITATION_MASK}', retrieve relevant documents that address and/or describe the citation\n<|embed|>\n"
    retrieval_instruction_document = f"<|embed|>\n"
    reranking_instruction = lambda q, d: f"You are given a query with a citation marked by '{CITATION_MASK}' and a paragraph. A paragraph is relevant if it addresses, describes and/or contains information about the citation. A paragraph is not relevant if it doesn't contain information about the citation, even if it mentions similar topics. Is the paragraph below relevant to the query below? The answer should be 'Relevance score: X' where X is a number from 0-10. 0 means completely irrelevant, 10 means highly relevant and completely addresses the query. Don't output anything else. Here is the query:<start_query>{q}<end_query>Here is the paragraph:<start_paragraph>{d}<end_paragraph>"


    evaluation_records = {}
    queries = citing_sents if not EXPAND_QUERY_CONTEXT else citing_context
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
    queries = [list(itertools.repeat(q, len(target_doc_sections))) for q in queries] # repeat queries for each candidate doc
    queries = list(itertools.chain.from_iterable(queries)) # flatten
    query_idx = [list(itertools.repeat(i, len(target_doc_sections))) for i in range(num_queries)] # repeat query idx for each candidate doc (for later access via the index in queries)
    query_idx = list(itertools.chain.from_iterable(query_idx)) # flatten

    # separate section labels and documents
    doc_lbls = [d.split(LABEL_SEPARATOR)[0] for d in target_doc_sections]
    docs = [d.split(LABEL_SEPARATOR)[1] for d in target_doc_sections]

    document_labels = list(itertools.repeat(doc_lbls, num_queries))
    document_labels = list(itertools.chain.from_iterable(document_labels)) # flatten
    documents = list(itertools.repeat(docs, num_queries))
    documents = list(itertools.chain.from_iterable(documents)) # flatten

    assert len(queries) == len(document_labels) == len(documents), f"{len(queries)}, {len(document_labels)}, {len(documents)}"

    print() # for console progress report

    num_batch = 1
    for i in range(0, len(queries), BATCH_SIZE):

        sys.stdout.write("\033[F")
        print(f"[RETRIEVAL] processing batch {num_batch}/{(len(queries)//BATCH_SIZE)+1}")
        num_batch += 1

        query_inputs = queries[i:i+BATCH_SIZE]
        document_inputs = documents[i:i+BATCH_SIZE]
        # score aggregation for simple self-consistency, see https://arxiv.org/abs/2505.12570 p.3 chapter 3
        scores_for_each_llm_call = [] # num_llm_calls x batch_size
        for _ in range(M_RETR):
            query_embs = retriever.encode(query_inputs, instruction=retrieval_instruction_query)
            doc_embs = retriever.encode(document_inputs, instruction=retrieval_instruction_document)
            scores_for_each_llm_call.append([query_embs[j] @ doc_embs[j] for j in range(len(query_inputs))])

        batch_sim_scores = [
            stat.mean([llm_call_scores[j] for llm_call_scores in scores_for_each_llm_call])
            for j in range(len(query_inputs)) 
        ]

        # update eval data with the current batch
        query_index = None
        for j in range(len(query_inputs)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE

            queries_idx = i+j # aka index within the list 'queries'
            query_index = query_idx[queries_idx]

            retrieved_documents = evaluation_records[f"query-{query_index}"]["retrieved documents"]
            retrieved_documents[document_labels[queries_idx]] = {
                "section chunk": documents[queries_idx],
                "retrieval score": f"{batch_sim_scores[j]}",
            }
        evaluation_records[f"query-{query_index}"]["retrieved documents"] = retrieved_documents
    
    # retain only the top k retrieved documents
    for i in range(num_queries):
        retrieved_documents = evaluation_records[f"query-{i}"]["retrieved documents"]
        retrieved_documents = dict(sorted(retrieved_documents.items(), key=lambda item: item[1]["retrieval score"], reverse=True)[:TOPK_RETR])
        evaluation_records[f"query-{i}"]["retrieved documents"] = retrieved_documents


    # --- RERANKING ---
    reranking_inputs = []
    document_labels = []
    documents = []
    for i in range(num_queries):
        for doc_label, doc_dict in evaluation_records[f"query-{i}"]["retrieved documents"].items():
            q = evaluation_records[f"query-{i}"][f"query-{i}"]
            d = doc_dict["section chunk"]
            messages = [
                {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
                {"role": "user", "content": reranking_instruction(q, d)}
            ]
            reranking_inputs.append(messages)
            document_labels.append(doc_label)
            documents.append(d)

    query_idx = [list(itertools.repeat(i, TOPK_RETR)) for i in range(num_queries)] # repeat query idx for each topk retrieved doc (for later access via the index in reranking_inputs)
    query_idx = list(itertools.chain.from_iterable(query_idx)) # flatten

    assert len(reranking_inputs) == num_queries*TOPK_RETR, f"{len(reranking_inputs)}, {num_queries*TOPK_RETR}"

    print() # for console progress report

    num_batch = 1
    for i in range(0, len(reranking_inputs), BATCH_SIZE):

        sys.stdout.write("\033[F")
        print(f"[RERANKING] processing batch {num_batch+1}/{(len(reranking_inputs)//BATCH_SIZE)+1}")
        num_batch += 1

        batch_reranking_inputs = reranking_inputs[i:i+BATCH_SIZE]

        # score aggregation for simple self-consistency, see https://arxiv.org/abs/2505.12570 p.3 chapter 3
        scores_for_each_llm_call = [] # num_llm_calls x batch_size
        for _ in range(M_RERA):
            batch_scores = []
            for messages in batch_reranking_inputs:
                chat = rera_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = rera_tokenizer([chat], return_tensors="pt").to("cuda")
                generated_encoded_tokens = reranker.generate(**inputs, max_new_tokens=512)
                generated_encoded_tokens = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_encoded_tokens)
                ]
                response = rera_tokenizer.batch_decode(generated_encoded_tokens, skip_special_tokens=True)[0]
            
                try:
                    inst_score = float(response.lstrip("Relevance score: ").rstrip("<|im_end|>"))*0.1
                except ValueError as e:
                    print(response)
                    raise e
                batch_scores.append(inst_score)
            scores_for_each_llm_call.append(batch_scores)

        batch_aggr_scores = [
            stat.mean([llm_call_scores[j] for llm_call_scores in scores_for_each_llm_call])
            for j in range(len(batch_reranking_inputs)) 
        ]

        # update eval data with the current batch
        query_index = None
        for j in range(len(batch_reranking_inputs)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE

            reranking_inputs_idx = i+j # aka index within the list 'reranking_inputs'
            query_index = query_idx[reranking_inputs_idx]

            reranked_documents = evaluation_records[f"query-{query_index}"]["reranked documents"]
            reranked_documents[document_labels[reranking_inputs_idx]] = {
                "section chunk": documents[reranking_inputs_idx],
                "reranking score": f"{batch_aggr_scores[j]}",
            }
        reranked_documents = evaluation_records[f"query-{query_index}"]["reranked documents"]
        
    # sort reranked docs
    for i in range(num_queries):
        reranked_documents = evaluation_records[f"query-{i}"]["reranked documents"]
        reranked_documents = dict(sorted(reranked_documents.items(), key=lambda item: item[1]["reranking score"], reverse=True)[:TOPK_RERA])
        evaluation_records[f"query-{i}"]["reranked documents"] = reranked_documents

        # obtain final ranking score s
        # s = (1-delta)*s_retr + delta*s_rera
        final_scores = {}
        # reranked_documents still exists, retrieved_documents does not
        retrieved_documents = evaluation_records[f"query-{i}"]["retrieved documents"]
        for label in reranked_documents:
            doc = reranked_documents[label]["section chunk"]
            s_retr = float(retrieved_documents[label]["retrieval score"])
            s_rera = float(reranked_documents[label]["reranking score"])
            s = (1-DELTA)*s_retr + DELTA*s_rera
            final_scores[label] = {
                "section chunk": doc,
                "final ranking score": s
            }
        final_scores = dict(sorted(final_scores.items(), key=lambda item: item[1]["final ranking score"], reverse=True))

        evaluation_records[f"query-{i}"]["final ranking"] = final_scores


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
            "reranker": RERANKER_MODEL,
            "m_retrieval": M_RETR,
            "m_reranking": M_RERA,
            "final_score_delta": DELTA,
            "target_chunking": ENABLE_CHUNKING,
            "chunk_size": CHUNK_SIZE,
            "expand_query": EXPAND_QUERY_CONTEXT,
            "query_context": QUERY_CONTEXT_SIZE,
            "retriever_topk": TOPK_RETR,
            "reranker_topk": TOPK_RERA,
            "instructions": {
                "retr_query": retrieval_instruction_query,
                "retr_document": retrieval_instruction_document,
                "rera_scoring": reranking_instruction("", "")
            }
        },
        "records": evaluation_records
    }

    with open('out/evaluation_records.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    