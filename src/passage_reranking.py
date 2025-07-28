import sys
import traceback
import statistics as stat
import random
import time
from util import minmax_normalization
from passage_retrieval_instructions import *


# currently works only for eval, no batch processing
def reranking_and_scoring(
    evaluation_records, 
    generated_context, 
    document_labels,
    documents,
    reference_ids,
    reranker,
    rera_tokenizer,
    config,
    silent=False
):
    PASSAGES_PER_CALL = config["rera_passages_per_call"]
    SC_PERMUTATION_MODE = config["rera_sc_permutation_mode"]
    M_RERA = config["m_reranking"]
    RERA_TEMPERATURE = config["reranker_temperature"]
    TOPK_RERA = config["reranker_topk"]
    DELTA = config["final_score_delta"]
    NORMALIZE_SCORES = config["ranking_score_normalization"]

    if SC_PERMUTATION_MODE not in ["stb", "bts"]:
        raise NotImplementedError(f"self-consistency permutation mode {SC_PERMUTATION_MODE} is not implemented. Currently supported are: stb, bts")

    if PASSAGES_PER_CALL > TOPK_RERA and not silent:
        print(f"[RERANKING] self-consistency batch size ({PASSAGES_PER_CALL}) cannot be larger that reranker topk ({TOPK_RERA}). Setting PASSAGES_PER_CALL={TOPK_RERA}")
        PASSAGES_PER_CALL = TOPK_RERA

    if not silent:
        print() # for console progress report

    zip_documents = list(zip(document_labels, documents))

    sc_batching_timecost = 0
    generation_timecost = 0
    response_parsing_timecost = 0
    scores_for_each_llm_call = [] # num_llm_calls x batch_size, contains (label, document, score) triples
    for m in range(M_RERA): # iterates llm calls

        if not silent:
            sys.stdout.write("\033[F")
            print(f"[RERANKING] reranking {len(documents)} passages - self-consistency call {m+1}/{M_RERA}")

        start = time.time()
        # shuffles before each llm call -> passage mixture of each batch is different across llm calls
        if SC_PERMUTATION_MODE == "stb":
            random.shuffle(zip_documents)

        # create self-consistency batches
        selfconsist_batched_inputs = [] # sc_batch_size/PASSAGES_PER_CALL x 1
        selfconsist_passages_lengths = [] # for checking if enough scores are generated
        shuffled_batch_labels = []
        shuffled_zip_documents = []
        for selfconsist_passages_idx in range(0, len(documents), PASSAGES_PER_CALL): # iterates self-consistency batches

            selfconsist_passages_documents = None
            terminate_after_this_batch = False
            if len(documents)-(selfconsist_passages_idx+PASSAGES_PER_CALL) >= 3:
                selfconsist_passages_documents = zip_documents[selfconsist_passages_idx:selfconsist_passages_idx+PASSAGES_PER_CALL]
                selfconsist_passages_lengths.append(len(selfconsist_passages_documents))
            else: # append the last batch if it has < 3 passages - this reduces llm parsing errors
                selfconsist_passages_documents = zip_documents[selfconsist_passages_idx:len(documents)]
                selfconsist_passages_lengths.append(len(selfconsist_passages_documents))
                terminate_after_this_batch = True

            # shuffles after batching -> passage mixture of each batch is the same across llm calls
            if SC_PERMUTATION_MODE == "bts":
                random.shuffle(selfconsist_passages_documents)
            
            for (l, d) in selfconsist_passages_documents:
                shuffled_batch_labels.append(l)
                shuffled_zip_documents.append(d)

            selfconsist_batched_inputs.append(rera_tokenizer.apply_chat_template([
                    {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
                    {"role": "user", "content": reranking_instruction(generated_context, [d for (l, d) in selfconsist_passages_documents])}
                ],
                tokenize=False,
                add_generation_prompt=True
            ))
            if terminate_after_this_batch:
                break
        sc_batching_timecost += time.time() - start

        start = time.time()
        # feeding every batch into the model at once consimes more memory per gpu
        # thats why the batches are looped here
        responses = []
        for batch_input in selfconsist_batched_inputs:
            llm_call_input = rera_tokenizer([batch_input], return_tensors="pt", padding=True, padding_side="left").to(config["reranker_device"])
            generated_encoded_tokens = reranker.generate(
                **llm_call_input, 
                max_new_tokens=128,
                temperature=RERA_TEMPERATURE
            )
            responses.append(rera_tokenizer.batch_decode(generated_encoded_tokens, skip_special_tokens=True)[0])
        generation_timecost += time.time() - start

        start = time.time()
        batch_scores = []
        for selfconsist_passages_idx in range(len(responses)): # iterates self-consistency batches
            try:
                original_llm_chatlog = responses[selfconsist_passages_idx]
                response = responses[selfconsist_passages_idx].split("assistant\n")[-1] # list only
                response = response.lstrip("[").rstrip("]")
                if len(response.split(", ")) == selfconsist_passages_lengths[selfconsist_passages_idx]:
                    split_str = ", "
                else:
                    split_str = "," # happens sometimes
                selfconsist_passages_scores = [float(score)*0.1 for score in response.split(split_str)]
                assert len(selfconsist_passages_scores) == selfconsist_passages_lengths[selfconsist_passages_idx]
            except Exception as e:
                raise InvalidLLMResponseError(f"reranker response could not be parsed successfully.\n\nLLM chatlog:{original_llm_chatlog}\n\nTraceback:{traceback.format_exc()}\n")
            [batch_scores.append(s) for s in selfconsist_passages_scores]

        scores_for_each_llm_call.append(zip(shuffled_batch_labels, shuffled_zip_documents, batch_scores))
        response_parsing_timecost += time.time() - start

    # print("sc batching cost (time): ", sc_batching_timecost)
    # print("generation cost (time): ", generation_timecost)
    # print("response parsing cost (time): ", response_parsing_timecost)
    
    start = time.time()
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
        try:
            assert len(scores) == M_RERA # ensure that we have all the scores
        except AssertionError:
            raise InvalidLLMResponseError(f"something went wrong during score collection.\n\nllm_call_data:{llm_call_data}")
        batch_rera_scores[label] = stat.mean(scores)


    # update eval data with the current batch
    reranked_documents = evaluation_records[f"reference_ids-{reference_ids}"]["reranked documents"]
    for label in batch_rera_scores:
        reranked_documents[label] = {
            "section chunk": batch_rera_docs[label],
            "reranking score": float(batch_rera_scores[label]),
        }
    evaluation_records[f"reference_ids-{reference_ids}"]["reranked documents"] = reranked_documents
        

    # sort reranked docs
    reranked_documents = evaluation_records[f"reference_ids-{reference_ids}"]["reranked documents"]
    
    if NORMALIZE_SCORES: # min-max normalization
        scores = [float(reranked_documents[section_id]["reranking score"]) for section_id in reranked_documents]
        scores = minmax_normalization(scores)
        for idx, section_id in enumerate(reranked_documents): # iterates doc records
            reranked_documents[section_id]["reranking score"] = scores[idx]

    reranked_documents = dict(sorted(reranked_documents.items(), key=lambda item: item[1]["reranking score"], reverse=True)[:TOPK_RERA])
    evaluation_records[f"reference_ids-{reference_ids}"]["reranked documents"] = reranked_documents
    # print("normalization cost (time): ", time.time() - start)

    start = time.time()
    # obtain final ranking score s
    # s = (1-delta)*s_retr + delta*s_rera
    final_scores = {}
    # reranked_documents still exists, retrieved_documents does not
    if config["enable_passage_retriever"]:
        retrieved_documents = evaluation_records[f"reference_ids-{reference_ids}"]["retrieved documents"]
    for label in reranked_documents:
        doc = reranked_documents[label]["section chunk"]

        if config["enable_passage_retriever"]:
            s_retr = float(retrieved_documents[label]["retrieval score"])
            s_rera = float(reranked_documents[label]["reranking score"])
            s = (1-DELTA)*s_retr + DELTA*s_rera
        else: # just use the reranking score
            s = float(reranked_documents[label]["reranking score"])

        final_scores[label] = {
            "section chunk": doc,
            "final ranking score": s
        }
    final_scores = dict(sorted(final_scores.items(), key=lambda item: item[1]["final ranking score"], reverse=True))
    evaluation_records[f"reference_ids-{reference_ids}"]["final ranking"] = final_scores
    # print("final scoring cost (time): ", time.time() - start)

    ranked_passages = []
    ranked_passage_labels = []
    ranked_passage_scores = []
    for label in final_scores:
        ranked_passages.append(final_scores[label]["section chunk"])
        ranked_passage_labels.append(label)
        ranked_passage_scores.append(final_scores[label]["final ranking score"])

    return {
        "ranked_passages": ranked_passages, 
        "ranked_passage_labels": ranked_passage_labels, 
        "ranked_passage_scores": ranked_passage_scores, 
        "final_scores": final_scores
    }


class InvalidLLMResponseError(Exception):
    """LLM response does not meet set requirements or was otherwise unsuccessfully parsed."""
    pass