import sys
import statistics as stat
import random
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
    PASSAGES_PER_CALL = config["rera_passages_per_call"]
    SC_PERMUTATION_MODE = config["rera_sc_permutation_mode"]
    M_RERA = config["m_reranking"]
    RERA_TEMPERATURE = config["reranker_temperature"]
    TOPK_RERA = config["reranker_topk"]
    DELTA = config["final_score_delta"]
    NORMALIZE_SCORES = config["ranking_score_normalization"]

    # ensure that inputs are not batched during evaluation
    if (not reranker.training) and (BATCH_SIZE != TOPK_RERA):
        raise NotImplementedError("[RERANKING] weight update batch size is irrelevant for evaluation, don't change it!")

    if SC_PERMUTATION_MODE not in ["stb", "bts"]:
        raise NotImplementedError(f"self-consistency permutation mode {SC_PERMUTATION_MODE} is not implemented. Currently supported are: stb, bts")

    if BATCH_SIZE > TOPK_RERA:
        print(f"[RERANKING] batch size ({BATCH_SIZE}) cannot be larger that reranker topk ({TOPK_RERA}). Setting BATCH_SIZE={TOPK_RERA}")
        BATCH_SIZE = TOPK_RERA

    if PASSAGES_PER_CALL > TOPK_RERA:
        print(f"[RERANKING] self-consistency batch size ({PASSAGES_PER_CALL}) cannot be larger that reranker topk ({TOPK_RERA}). Setting PASSAGES_PER_CALL={TOPK_RERA}")
        PASSAGES_PER_CALL = TOPK_RERA

    print() # for console progress report

    for k in range(num_queries): # iterates queries
        num_batch = 1
        for i in range(0, len(documents[k]), BATCH_SIZE): # iterates sections, batched

            sys.stdout.write("\033[F")
            print(f"[RERANKING] processing query {k+1}/{num_queries} - batch {num_batch}/{(len(documents[k])//BATCH_SIZE)+1}")

            batch_query = single_queries[k]

            l = document_labels[k][i:i+BATCH_SIZE]
            d = documents[k][i:i+BATCH_SIZE]
            batch_documents = list(zip(l, d))

            scores_for_each_llm_call = [] # num_llm_calls x batch_size, contains (label, document, score) triples
            for _ in range(M_RERA): # iterates llm calls

                # shuffles before each llm call -> passage mixture of each batch is different across llm calls
                if SC_PERMUTATION_MODE == "stb":
                    random.shuffle(batch_documents)

                # create self-consistency batches
                batch_reranking_input = [] # batch_size/PASSAGES_PER_CALL x 1
                sc_passages_lengths = [] # for checking if enough scores are generated
                shuffled_batch_labels = []
                shuffled_batch_documents = []
                for sc_passages_idx in range(0, BATCH_SIZE, PASSAGES_PER_CALL): # iterates self-consistency batches

                    sc_passages_documents = batch_documents[sc_passages_idx:sc_passages_idx+PASSAGES_PER_CALL]
                    sc_passages_lengths.append(len(sc_passages_documents))

                    # shuffles after batching -> passage mixture of each batch is the same across llm calls
                    if SC_PERMUTATION_MODE == "bts":
                        random.shuffle(sc_passages_documents)
                    
                    for (l, d) in sc_passages_documents:
                        shuffled_batch_labels.append(l)
                        shuffled_batch_documents.append(d)

                    batch_reranking_input.append(rera_tokenizer.apply_chat_template([
                            {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
                            {"role": "user", "content": reranking_instruction(batch_query, [d for (l, d) in sc_passages_documents])}
                        ],
                        tokenize=False,
                        add_generation_prompt=True
                    ))

                llm_call_input = rera_tokenizer(batch_reranking_input, return_tensors="pt", padding=True, padding_side="left").to("cuda")
                generated_encoded_tokens = reranker.generate(
                    **llm_call_input, 
                    max_new_tokens=128,
                    temperature=RERA_TEMPERATURE
                )
                responses = rera_tokenizer.batch_decode(generated_encoded_tokens, skip_special_tokens=True)

                batch_scores = []
                for sc_passages_idx in range(len(responses)): # iterates self-consistency batches
                    try:
                        response = responses[sc_passages_idx].split("assistant\nRelevance scores: ")[1] # list only
                        response = response.lstrip("[").rstrip("]")
                        sc_passages_scores = [float(score)*0.1 for score in response.split(", ")]
                        assert len(sc_passages_scores) == sc_passages_lengths[sc_passages_idx]
                    except AssertionError as e:
                        print(f"[RERANKING] generated scores don't match current sc batch size: {len(sc_passages_scores)} != {sc_passages_lengths[sc_passages_idx]}")
                        print(responses[sc_passages_idx])
                        raise e
                        # sc_passages_scores = sc_passages_scores[:len(batch_documents)] # dirty fix ;)
                        # [sc_passages_scores.append(0) for _ in range(len(batch_documents)-len(sc_passages_scores))] # zero padding
                    except Exception as e:
                        print(batch_reranking_input)
                        print(response)
                        raise e
                    [batch_scores.append(s) for s in sc_passages_scores]

                scores_for_each_llm_call.append(zip(shuffled_batch_labels, shuffled_batch_documents, batch_scores))
            

            batch_rera_scores = {} # batch_rera_scores[label] = score 
            batch_rera_docs = {} # batch_rera_docs[label] = doc 
            for llm_call_data in scores_for_each_llm_call: # iterates llm calls
                for (label, document, score) in llm_call_data: # iterates batch
                    try:
                        batch_rera_scores[label].append(score)
                    except KeyError:
                        batch_rera_scores[label] = [score]
                        batch_rera_docs[label] = document


            for label in batch_rera_scores:
                scores = batch_rera_scores[label]
                assert len(scores) == M_RERA # ensure that we have all the scores
                batch_rera_scores[label] = stat.mean(scores)


            # update eval data with the current batch
            reranked_documents = evaluation_records[f"query-{k}"]["reranked documents"]
            for label in batch_rera_scores:
                reranked_documents[label] = {
                    "section chunk": batch_rera_docs[label],
                    "reranking score": float(batch_rera_scores[label]),
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