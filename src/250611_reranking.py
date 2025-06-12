import sys
import statistics as stat
from util import minmax_normalization


def reranking_and_scoring(
    evaluation_records, 
    num_queries, 
    single_queries, 
    document_labels,
    documents,
    reranking_instruction,
    reranker,
    rera_tokenizer,
    config
):
    BATCH_SIZE = config["rera_batch_size"]
    M_RERA = config["m_reranking"]
    RERA_TEMPERATURE = config["reranker_temperature"]
    TOPK_RERA = config["reranker_topk"]
    DELTA = config["final_score_delta"]
    NORMALIZE_SCORES = config["ranking_score_normalization"]

    if BATCH_SIZE > TOPK_RERA:
        print(f"[RERANKING] batch size ({BATCH_SIZE}) cannot be larger that reranker topk ({TOPK_RERA}). Setting batch_size={TOPK_RERA}")
        BATCH_SIZE = TOPK_RERA

    print() # for console progress report

    for k in range(num_queries): # iterates queries
        num_batch = 1
        for i in range(0, len(documents[k]), BATCH_SIZE): # iterates sections, batched

            sys.stdout.write("\033[F")
            print(f"[RERANKING] processing query {k+1}/{num_queries} - batch {num_batch}/{(len(documents[k])//BATCH_SIZE)+1}")

            batch_query = single_queries[k]
            batch_documents = documents[k][i:i+BATCH_SIZE]

            # score aggregation for batched self-consistency
            # https://arxiv.org/abs/2505.12570
            scores_for_each_llm_call = [] # num_llm_calls x batch_size
            for _ in range(M_RERA): # iterates llm calls
                batch_reranking_input = rera_tokenizer.apply_chat_template([
                        {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
                        {"role": "user", "content": reranking_instruction(batch_query, batch_documents)}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                llm_call_input = rera_tokenizer(batch_reranking_input, return_tensors="pt", padding=True, padding_side="left").to("cuda")
                generated_encoded_tokens = reranker.generate(
                    **llm_call_input, 
                    max_new_tokens=128,
                    temperature=RERA_TEMPERATURE
                )
                response = rera_tokenizer.batch_decode(generated_encoded_tokens, skip_special_tokens=True)

                batch_scores = None
                try:
                    response = response[0].split("assistant\nRelevance scores: ")[1] # list only
                    response = response.lstrip("[").rstrip("]")
                    batch_scores = [float(score)*0.1 for score in response.split(", ")]
                    assert len(batch_scores) == len(batch_documents)
                except AssertionError as e:
                    print(f"[RERANKING] generated scores don't match current batch size: {len(batch_scores)} != {len(batch_documents)}")
                    batch_scores = batch_scores[:len(batch_documents)] # dirty fix ;)
                    [batch_scores.append(0) for _ in range(len(batch_documents)-len(batch_scores))] # zero padding
                except Exception as e:
                    print(batch_reranking_input)
                    print(response)
                    raise e
                scores_for_each_llm_call.append(batch_scores)
            
            batch_rera_scores = []
            for j in range(len(batch_documents)): # iterates current batch
                mean_over_llm_calls = stat.mean([llm_call_scores[j] for llm_call_scores in scores_for_each_llm_call]) # iterates llm calls
                batch_rera_scores.append(mean_over_llm_calls)

            # update eval data with the current batch
            reranked_documents = evaluation_records[f"query-{k}"]["reranked documents"]
            for j in range(len(batch_documents)): # can't use BATCH_SIZE here bc the last batch might be shorter than BATCH_SIZE
                global_batch_idx = ((num_batch-1)*BATCH_SIZE)+j
                reranked_documents[document_labels[k][global_batch_idx]] = {
                    "section chunk": documents[k][global_batch_idx],
                    "reranking score": float(batch_rera_scores[j]),
                }
            evaluation_records[f"query-{k}"]["reranked documents"] = reranked_documents

            num_batch += 1
            

        # sort reranked docs
        reranked_documents = evaluation_records[f"query-{k}"]["reranked documents"]
        
        if NORMALIZE_SCORES: # min-max normalization
            scores = [float(reranked_documents[section_id]["reranking score"]) for section_id in reranked_documents]
            scores = minmax_normalization(scores)
            for idx, section_id in enumerate(reranked_documents): # iterates doc records
                reranked_documents[section_id]["reranking score"] = scores[idx]

        reranked_documents = dict(sorted(reranked_documents.items(), key=lambda item: item[1]["reranking score"], reverse=True)[:TOPK_RERA])
        evaluation_records[f"query-{k}"]["reranked documents"] = reranked_documents


        # obtain final ranking score s
        # s = (1-delta)*s_retr + delta*s_rera
        final_scores = {}
        # reranked_documents still exists, retrieved_documents does not
        retrieved_documents = evaluation_records[f"query-{k}"]["retrieved documents"]
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
        evaluation_records[f"query-{k}"]["final ranking"] = final_scores