import sys
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np

from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index
from evaluation_generation_instruction import judge_instruction, judge_instruction2
from scholarcopilot_generation import stream_generate
from dataset_loaders import load_generation_eval_dataset, arxiv_to_corpus_id, scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset, load_scholarcopilot_metadata_corpus


def load_generation_evaluation_model(config):
    judge_config = AutoConfig.from_pretrained(config["judge"])
    judge = AutoModelForCausalLM.from_pretrained(config["judge"], config=judge_config)
    judge.to(config["judge_device"])

    judge_tokenizer = AutoTokenizer.from_pretrained(config["judge"])
    judge_tokenizer.add_tokens(config["special_tokens"])
    # judge.resize_token_embeddings(len(judge_tokenizer))
    judge.generation_config.pad_token_id = judge_tokenizer.pad_token_id
    print("judge loaded.")
    return judge, judge_tokenizer


def stream_generate_discretizer(
        input_context, 
        index, 
        lookup_indices, 
        model, 
        tokenizer, 
        retrieval_dataset,
        sc_metadata_corpus,
        arxiv_to_corpus_id_map, 
        passage_retrieval_models, 
        config, 
        generation_breakpoint=15000, 
        do_passage_retrieval=True,
        catch_retrieval_fails=True,
        silent=False,
        save_passage_records=False
):
    gen = stream_generate(
        input_context, 
        [], 
        index, 
        lookup_indices, 
        model, 
        tokenizer, 
        retrieval_dataset,
        sc_metadata_corpus,
        None, # passages_data
        arxiv_to_corpus_id_map, 
        passage_retrieval_models, 
        config, 
        generation_breakpoint=generation_breakpoint, 
        do_passage_retrieval=do_passage_retrieval,
        catch_retrieval_fails=catch_retrieval_fails,
        silent=silent,
        save_passage_records=save_passage_records
    )
    for t in gen:
        generated_paper, citations_data, fails = t
    return generated_paper, citations_data, fails


def apply_generation_breakpoint(gold_generation, generation_breakpoint, tokenizer, config):
    gold_generation = tokenizer(gold_generation).to(config["scholarcopilot_device"])
    gold_generation = tokenizer.decode(gold_generation["input_ids"][:generation_breakpoint+1])
    gold_generation = gold_generation.replace(config["tokenizer_begin_token"], "")
    return gold_generation


def parse_judge_response(response):
    scores = {
        "Content Relevance": -1,
        "Logical Coherence": -1,
        "Academic Rigor": -1,
        "Background Completeness": -1,
        "Innovation Statement": -1
    }
    try:
        response = response.split("<｜Assistant｜>")[1]
        total = -1
        for score_label in scores:
            score = response.split(score_label)[-1]
            score = score.split(" ")[1].split("/")[0]
            score = score.lstrip("<").rstrip(">")
            scores[score_label] = float(score)
            total += float(score)
        scores["total"] = total
        for label, score in scores.items():
            if label == "total":
                assert score >= 0 and score <= 25
            else:
                assert score >= 0 and score <= 5
    except Exception:
        # print(response)
        raise ResponseParsingError
    return scores


class ResponseParsingError(Exception):
    """LLM response couldn't be parsed"""
    pass


if __name__ == "__main__":
    config = get_config("cfg/config_thesis_eval_generation.json")

    model_path = "scholarcopilot_model_v1208/"
    model, tokenizer = load_model(model_path, config)

    passage_retrieval_models = get_passage_retrieval_models(config)

    judge, judge_tokenizer = load_generation_evaluation_model(config)

    docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    docs_corpus_id_map = arxiv_to_corpus_id(docs_id_map_path, processed_corpus_path)
    docs_arxiv_id_map = {v: k for k, v in docs_corpus_id_map.items()}

    sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
    sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)
    sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
    overlap = 0
    for key in sc_corpus_id_map:
        if key in docs_corpus_id_map:
            overlap += 1
    sc_percentage = "{:.2f}".format((overlap/len(sc_corpus_id_map.items()))*100)
    print(f"arxiv_id mappings loaded. dataset overlap is {overlap}, {sc_percentage}% of the retrieval corpus")

    index_dir = "scholarcopilot_data/index"
    lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")

    corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_metadata_corpus = load_scholarcopilot_metadata_corpus(corpus_path)
    
    docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

    shuffle = True
    eval_dataset_path = "data_thesis/eval_dataset_generation.jsonl"
    eval_dataset, eval_indices = load_generation_eval_dataset(
        eval_dataset_path, 
        docs_retrieval_dataset, 
        sc_arxiv_id_map, 
        max_samples=100000, 
        shuffle=shuffle
    )

    samples = 0
    max_samples = 10
    retrieval_fails = 0
    llm_fails = 0
    catch_retrieval_fails = True
    total_num_retrievals = 0
    generation_breakpoint = 30000
    shuffle_instruction = True
    eps = 1e-6 # fail metrics
    scoring_records = []
    all_sc_judge_scores_avg = {
        "Content Relevance": [],
        "Logical Coherence": [],
        "Academic Rigor": [],
        "Background Completeness": [],
        "Innovation Statement": [],
        "total": []
    }
    all_sc_pr_judge_scores_avg = {
        "Content Relevance": [],
        "Logical Coherence": [],
        "Academic Rigor": [],
        "Background Completeness": [],
        "Innovation Statement": [],
        "total": []
    }

    # override some model settings to match the settings defined in this file
    config["custom_save_dir"] = f"out_thesis/eval_generation_judge_instruction2{"-only-scores" if config["judge_should_only_score"] else ""}_{generation_breakpoint}_{max_samples}"

    print()
    for i in eval_indices:
        # sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        item = eval_dataset[i]
        context = item["context"]
        gold_generation = apply_generation_breakpoint(item["introduction"], generation_breakpoint, tokenizer, config)

        # generate with SC
        start = time.time()
        sc_generated_paper, sc_citations_data, sc_fails = stream_generate_discretizer(
            context, 
            index, 
            lookup_indices, 
            model, 
            tokenizer, 
            docs_retrieval_dataset,
            sc_metadata_corpus,
            docs_corpus_id_map, 
            passage_retrieval_models, 
            config,
            generation_breakpoint=generation_breakpoint,
            do_passage_retrieval=False,
            catch_retrieval_fails=catch_retrieval_fails,
            silent=True,
            save_passage_records=False
        )
        sc_generated_paper = item["introduction_start"]+" "+sc_generated_paper.replace(context, "")
        retrieval_fails += sc_fails[0]
        llm_fails += sc_fails[1]
        total_num_retrievals += len(sc_citations_data)
        print(f"gen1 done in {time.time() - start}")

        # generate with SC+PR
        start = time.time()
        sc_pr_generated_paper, sc_pr_citations_data, sc_pr_fails = stream_generate_discretizer(
            context, 
            index, 
            lookup_indices, 
            model, 
            tokenizer, 
            docs_retrieval_dataset,
            sc_metadata_corpus,
            docs_corpus_id_map, 
            passage_retrieval_models, 
            config,
            generation_breakpoint=generation_breakpoint,
            do_passage_retrieval=True,
            catch_retrieval_fails=catch_retrieval_fails,
            silent=True,
            save_passage_records=False
        )
        sc_pr_generated_paper = item["introduction_start"]+" "+sc_pr_generated_paper.replace(context, "")
        retrieval_fails += sc_pr_fails[0]
        llm_fails += sc_pr_fails[1]
        total_num_retrievals += len(sc_pr_citations_data)
        print(f"gen2 done in {time.time() - start}")

        # judge each generated output individually (prompt from SC paper)
        # (judge both at once by contrasting them?)
        judge_scores_avg = [] # [0]: SC, [1]: SC PR
        judge_responses = [] # [0]: SC, [1]: SC PR
        for gen in [sc_generated_paper, sc_pr_generated_paper]:
            start = time.time()
            scores_per_call = []
            scoring_fails = 0
            num_successful_scorings = 0
            per_model_responses = []
            print()
            while num_successful_scorings < config["m_judging"]: # try to score until enough scorings are collected
                sys.stdout.write("\033[F")
                print(f"got {num_successful_scorings} scorings after {num_successful_scorings+scoring_fails} tries")
                try:
                    chat = judge_tokenizer.apply_chat_template(
                        [{"role": "user", "content": judge_instruction2(item["title"], item["abstract"], gold_generation, gen, shuffle=shuffle_instruction, only_scores=config["judge_should_only_score"])}], 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    judge_input = judge_tokenizer([chat], return_tensors="pt", padding=True, padding_side="left").to(config["judge_device"])
                    res = judge.generate(
                        **judge_input,
                        max_new_tokens=generation_breakpoint,
                        temperature=config["judge_temperature"]
                    )
                    res = judge_tokenizer.batch_decode(res)[0]
                    scores_per_call.append(parse_judge_response(res))
                    per_model_responses.append(res.split("<｜Assistant｜>")[1])
                    num_successful_scorings += 1
                except ResponseParsingError:
                    scoring_fails += 1
                    continue

            scores_avg = {}
            for key in scores_per_call[0]: # initialization
                scores_avg[key] = 0
            for scores in scores_per_call: # iterates llm calls
                for key in scores: # iterates score categories
                    scores_avg[key] += scores[key]
            for key in scores_avg:
                scores_avg[key] /= len(scores_per_call)
            judge_scores_avg.append(scores_avg)
            judge_responses.append(per_model_responses)
            print(f"judging{len(judge_scores_avg)} done in {time.time() - start}")
            print(f"SC {"PR" if len(judge_scores_avg) == 2 else ""} scores:")
            print(scores_avg)

        scoring_records.append({
            "source_id": item["source_arxiv_id"],
            "gold": gold_generation,
            "sc_generation": sc_generated_paper,
            "sc_pr_generation": sc_pr_generated_paper,
            "sc_responses": judge_responses[0],
            "sc_pr_responses": judge_responses[1],
            "sc_scores": judge_scores_avg[0],
            "sc_pr_scores": judge_scores_avg[1]
        })

        for key in all_sc_judge_scores_avg:
            all_sc_judge_scores_avg[key].append(judge_scores_avg[0][key])
            all_sc_pr_judge_scores_avg[key].append(judge_scores_avg[1][key])
        
        samples += 1
        if samples >= max_samples:
            break
    
    sc_scores_stats = {}
    sc_pr_scores_stats = {}
    for key in all_sc_judge_scores_avg:
        sc_scores_stats[key] = {
            "mean": np.mean(all_sc_judge_scores_avg[key]),
            "std": np.std(all_sc_judge_scores_avg[key]),
            "var": np.var(all_sc_judge_scores_avg[key]),
            "scores": all_sc_judge_scores_avg[key]
        }
        sc_pr_scores_stats[key] = {
            "mean": np.mean(all_sc_pr_judge_scores_avg[key]),
            "std": np.std(all_sc_pr_judge_scores_avg[key]),
            "var": np.var(all_sc_pr_judge_scores_avg[key]),
            "scores": all_sc_pr_judge_scores_avg[key]
        }
        
    save_results({
        "eval_dataset": eval_dataset_path,
        "retrieval_index": index_dir,
        "num_samples": samples,
        "shuffled_samples": shuffle,
        "shuffled_instruction": shuffle_instruction,
        "generation_breakpoint": generation_breakpoint,
        "SC retrieval fails": retrieval_fails,
        "catch retrieval fails": catch_retrieval_fails,
        "PR LLM fails": llm_fails,
        "total fraction of fails": (retrieval_fails+llm_fails)/(total_num_retrievals+eps),
        "SC scores stats": sc_scores_stats,
        "SC PR scores stats": sc_pr_scores_stats,
        "scoring records": scoring_records
    }, config, mode="eval_generation")

