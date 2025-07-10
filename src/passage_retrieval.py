import sys
import statistics as stat
import numpy as np
import itertools
from util import minmax_normalization
from passage_retrieval_instructions import *


def retrieval(
    evaluation_records,
    generated_context, 
    document_labels,
    documents,
    reference_ids,
    retriever,
    config
):
    BATCH_SIZE = config["retr_batch_size"]
    M_RETR = config["m_retrieval"]
    QUERY_CONTEXT_SIZE = config["query_context"]
    CHUNK_SIZE = config["passage_length"]
    TOPK_RETR = config["retriever_topk"]
    NORMALIZE_SCORES = config["ranking_score_normalization"]
    CITATION_MASK_TOKEN = config["citation_mask_token"]

    print() # for console progress report

    num_batch = 1
    for i in range(0, len(documents), BATCH_SIZE): # iterates sections, batched

        sys.stdout.write("\033[F")
        print(f"[RETRIEVAL] retrieving {TOPK_RETR} out of {len(documents)} passages - batch {num_batch}/{(len(documents)//BATCH_SIZE)+1}")

        document_inputs = documents[i:i+BATCH_SIZE]
        query_inputs = list(itertools.repeat(generated_context, len(document_inputs)))

        # score aggregation for simple self-consistency, see https://arxiv.org/abs/2505.12570 p.3 chapter 3
        scores_for_each_llm_call = [] # num_llm_calls x batch_size
        for _ in range(M_RETR): # iterates llm calls
            query_embs = retriever.encode(
                query_inputs, 
                instruction=retrieval_instruction_query(CITATION_MASK_TOKEN), 
                batch_size=BATCH_SIZE, 
                max_length=QUERY_CONTEXT_SIZE
            )
            doc_embs = retriever.encode(
                document_inputs, 
                instruction=retrieval_instruction_document, 
                batch_size=BATCH_SIZE, 
                max_length=CHUNK_SIZE
            )

            llm_call_scores = []
            for j in range(len(query_inputs)): # iterates current batch
                s = query_embs[j] @ doc_embs[j] # mat. mult. aka dot product
                llm_call_scores.append(s)
            scores_for_each_llm_call.append(llm_call_scores)
        
        batch_sim_scores = []
        for j in range(len(query_inputs)): # iterates current batch
            mean_over_llm_calls = stat.mean([llm_call_scores[j] for llm_call_scores in scores_for_each_llm_call]) # iterates llm calls
            batch_sim_scores.append(mean_over_llm_calls)

        # update eval data with the current batch
        retrieved_documents = evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"]
        for j in range(len(query_inputs)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE
            float_score = float(batch_sim_scores[j])
            if np.isnan(float_score):
                float_score = -1
            global_batch_idx = ((num_batch-1)*BATCH_SIZE)+j
            retrieved_documents[document_labels[global_batch_idx]] = {
                "section chunk": documents[global_batch_idx],
                "retrieval score": float_score,
            }
        evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"] = retrieved_documents

        num_batch += 1


    # retain only the top k retrieved documents
    retrieved_documents = evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"]

    if NORMALIZE_SCORES: # min-max normalization
        scores = [float(retrieved_documents[section_id]["retrieval score"]) for section_id in retrieved_documents]
        scores = minmax_normalization(scores)
        for idx, section_id in enumerate(retrieved_documents): # iterates doc records
            retrieved_documents[section_id]["retrieval score"] = scores[idx]

    retrieved_documents = dict(sorted(retrieved_documents.items(), key=lambda item: item[1]["retrieval score"], reverse=True)[:TOPK_RETR])
    evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"] = retrieved_documents