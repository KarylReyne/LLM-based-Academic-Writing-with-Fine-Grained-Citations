import sys
import statistics as stat

def retrieval(
    evaluation_records, 
    num_queries, 
    queries, 
    document_labels,
    documents,
    retrieval_instruction_query,
    retrieval_instruction_document,
    retriever,
    config
):
    BATCH_SIZE = config["batch_size"]
    M_RETR = config["m_retrieval"]
    QUERY_CONTEXT_SIZE = config["query_context"]
    CHUNK_SIZE = config["chunk_size"]
    TOPK_RETR = config["retriever_topk"]

    print() # for console progress report

    for k in range(num_queries): # iterates queries
        # for min-max norm
        min_score = 1
        max_score = 0

        num_batch = 1
        for i in range(0, len(queries[k]), BATCH_SIZE): # iterates sections, batched

            sys.stdout.write("\033[F")
            print(f"[RETRIEVAL] processing query {k+1}/{len(queries)} - batch {num_batch}/{(len(queries[k])//BATCH_SIZE)+1}")

            query_inputs = queries[k][i:i+BATCH_SIZE]
            document_inputs = documents[k][i:i+BATCH_SIZE]

            # score aggregation for simple self-consistency, see https://arxiv.org/abs/2505.12570 p.3 chapter 3
            scores_for_each_llm_call = [] # num_llm_calls x batch_size
            for _ in range(M_RETR): # iterates llm calls
                query_embs = retriever.encode(query_inputs, instruction=retrieval_instruction_query, batch_size=BATCH_SIZE, max_length=QUERY_CONTEXT_SIZE)
                doc_embs = retriever.encode(document_inputs, instruction=retrieval_instruction_document, batch_size=BATCH_SIZE, max_length=CHUNK_SIZE)

                llm_call_scores = []
                for j in range(len(query_inputs)): # iterates current batch
                    s = query_embs[j] @ doc_embs[j]
                    llm_call_scores.append(s)
                scores_for_each_llm_call.append(llm_call_scores)
            
            batch_sim_scores = []
            for j in range(len(query_inputs)): # iterates current batch
                mean_over_llm_calls = stat.mean([llm_call_scores[j] for llm_call_scores in scores_for_each_llm_call]) # iterates llm calls
                batch_sim_scores.append(mean_over_llm_calls)
                min_score = min([min_score, mean_over_llm_calls])
                max_score = max([max_score, mean_over_llm_calls])

            # update eval data with the current batch
            retrieved_documents = evaluation_records[f"query-{k}"]["retrieved documents"]
            for j in range(len(query_inputs)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE
                global_batch_idx = ((num_batch-1)*BATCH_SIZE)+j
                retrieved_documents[document_labels[k][global_batch_idx]] = {
                    "section chunk": documents[k][global_batch_idx],
                    "retrieval score": f"{batch_sim_scores[j]}",
                }
            evaluation_records[f"query-{k}"]["retrieved documents"] = retrieved_documents

            num_batch += 1
        

        # retain only the top k retrieved documents
        retrieved_documents = evaluation_records[f"query-{k}"]["retrieved documents"]
        for key in retrieved_documents: # min-max normalization
            normalized_score = (float(retrieved_documents[key]["retrieval score"])-min_score)/(max_score-min_score)
            retrieved_documents[key]["retrieval score"] = f"{normalized_score}"
        retrieved_documents = dict(sorted(retrieved_documents.items(), key=lambda item: item[1]["retrieval score"], reverse=True)[:TOPK_RETR])
        evaluation_records[f"query-{k}"]["retrieved documents"] = retrieved_documents