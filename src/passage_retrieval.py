import sys
import statistics as stat
import torch
import numpy as np
import time
from util import minmax_normalization
from passage_retrieval_instructions import *


def retrieval(
    evaluation_records,
    generated_context, 
    document_labels,
    documents,
    reference_ids,
    retriever,
    tokenizer,
    config,
    silent=False
):
    BATCH_SIZE = config["retr_batch_size"]
    QUERY_CONTEXT_SIZE = config["query_context"]
    CHUNK_SIZE = config["passage_length"]
    TOPK_RETR = config["retriever_topk"]
    NORMALIZE_SCORES = config["ranking_score_normalization"]
    CITATION_MASK_TOKEN = config["citation_mask_token"]

    query_emb = retriever.encode(
        generated_context, 
        instruction=retrieval_instruction_query(CITATION_MASK_TOKEN), 
        batch_size=1,
        max_length=QUERY_CONTEXT_SIZE
    )

    if not silent:
        print() # for console progress report

    num_batch = 1
    encode_timecost = 0
    save_timecost = 0
    for i in range(0, len(documents), BATCH_SIZE): # iterates sections, batched
        if not silent:
            sys.stdout.write("\033[F")
            print(f"[RETRIEVAL] retrieving {TOPK_RETR} out of {len(documents)} passages - batch {num_batch}/{(len(documents)//BATCH_SIZE)+1}")

        document_inputs = documents[i:i+BATCH_SIZE]
        
        start = time.time()
        doc_embs = retriever.encode(
            document_inputs, 
            instruction=retrieval_instruction_document, 
            batch_size=BATCH_SIZE, 
            max_length=CHUNK_SIZE
        )
        encode_timecost += time.time() - start

        start = time.time()
        # update eval data with the current batch
        retrieved_documents = evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"]
        for j in range(len(document_inputs)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE

            s = query_emb @ doc_embs[j] # mat. mult. aka dot product

            if np.isnan(s):
                s = -1
            global_batch_idx = ((num_batch-1)*BATCH_SIZE)+j
            retrieved_documents[document_labels[global_batch_idx]] = {
                "section chunk": documents[global_batch_idx],
                "retrieval score": s,
            }
        evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"] = retrieved_documents
        save_timecost += time.time() - start

        num_batch += 1

    # print("encoding cost (time): ", encode_timecost)
    # print("score saving cost (time): ", save_timecost)

    start = time.time()
    # retain only the top k retrieved documents
    retrieved_documents = evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"]

    if NORMALIZE_SCORES: # min-max normalization
        scores = [float(retrieved_documents[section_id]["retrieval score"]) for section_id in retrieved_documents]
        scores = minmax_normalization(scores)
        for idx, section_id in enumerate(retrieved_documents): # iterates doc records
            retrieved_documents[section_id]["retrieval score"] = scores[idx]

    retrieved_documents = dict(sorted(retrieved_documents.items(), key=lambda item: item[1]["retrieval score"], reverse=True)[:TOPK_RETR])
    evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"] = retrieved_documents
    # print("normalization cost (time): ", time.time() - start)
    