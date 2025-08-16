import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from passage_retrieval_interface import get_config, get_passage_retrieval_models, save_results
from scholarcopilot_model import load_model, load_faiss_index, ScholarCopilotRetrievalError
from scholarcopilot_generation import stream_generate
from dataset_loaders import load_generation_eval_dataset, arxiv_to_corpus_id, scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset


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
        arxiv_to_corpus_id_map, 
        passage_retrieval_models, 
        config, 
        generation_breakpoint=15000, 
        do_passage_retrieval=True, 
        silent=False
):
    gen = stream_generate(
        input_context, 
        [], 
        index, 
        lookup_indices, 
        model, 
        tokenizer, 
        retrieval_dataset, 
        arxiv_to_corpus_id_map, 
        passage_retrieval_models, 
        config, 
        generation_breakpoint=generation_breakpoint, 
        do_passage_retrieval=do_passage_retrieval, 
        silent=silent
    )
    for t in gen:
        generated_paper, citations_data = t
    return generated_paper, citations_data


def apply_generation_breakpoint(gold_generation, generation_breakpoint, tokenizer, config):
    gold_generation = tokenizer(gold_generation).to(config["scholarcopilot_device"])
    gold_generation = tokenizer.decode(gold_generation["input_ids"][:generation_breakpoint+1])
    gold_generation = gold_generation.replace(config["tokenizer_begin_token"], "")
    return gold_generation


if __name__ == "__main__":
    config = get_config()

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

    index_dir = "scholarcopilot_data/index"
    lookup_indices_dir = "scholarcopilot_data/lookup_indices.npy"
    index, lookup_indices = load_faiss_index(index_dir, lookup_indices_dir)
    print("index building finished")
    
    docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

    shuffle = True
    eval_dataset_path = "data/eval_dataset_generation.jsonl"
    eval_dataset, eval_indices = load_generation_eval_dataset(
        eval_dataset_path, 
        docs_retrieval_dataset, 
        sc_arxiv_id_map, 
        max_samples=1000, 
        shuffle=shuffle
    )

    # print(eval_dataset[:3])

    samples = 0
    max_samples = 1
    retrieval_fails = 0
    generation_breakpoint = 15000
    print()
    for i in eval_indices:
        sys.stdout.write("\033[F")
        print(f"processing entry {samples}")

        try:
            item = eval_dataset[i]
            context = item["context"]
            gold_generation = apply_generation_breakpoint(item["introduction"], generation_breakpoint, tokenizer, config)

            # generate with SC
            sc_generated_paper, _ = stream_generate_discretizer(
                context, 
                index, 
                lookup_indices, 
                model, 
                tokenizer, 
                docs_retrieval_dataset, 
                docs_arxiv_id_map, 
                passage_retrieval_models, 
                config,
                generation_breakpoint=generation_breakpoint,
                do_passage_retrieval=False,
                silent=True
            )
            sc_generated_paper = item["introduction_start"]+" "+sc_generated_paper.replace(context, "")
            print("gen1 done")

            # generate with SC+PR
            sc_pr_generated_paper, _ = stream_generate_discretizer(
                context, 
                index, 
                lookup_indices, 
                model, 
                tokenizer, 
                docs_retrieval_dataset, 
                docs_arxiv_id_map, 
                passage_retrieval_models, 
                config,
                generation_breakpoint=generation_breakpoint,
                do_passage_retrieval=True, 
                silent=True
            )
            sc_pr_generated_paper = item["introduction_start"]+" "+sc_pr_generated_paper.replace(context, "")
            print("gen2 done")

            # judge each generated output individually (prompt from SC paper)
            judge_instruction = lambda gen: f"""
                You are a senior computer science scholar. Please evaluate the AI-generated content
                using the ground truth as reference.
                Evaluate the following five dimensions by comparing the AI-generated content with the
                ground truth:
                [Detailed Evaluation]
                1. Content Relevance:
                - Key strengths:
                - Main gaps:
                - Comparison with ground truth:
                2. Logical Coherence:
                - Key strengths:
                - Main gaps:
                - Comparison with ground truth:
                3. Academic Standards:
                - Key strengths:
                - Main gaps:
                - Comparison with ground truth:
                4. Background Completeness:
                - Key strengths:
                - Main gaps:
                - Comparison with ground truth:
                5. Innovation Statement:
                - Key strengths:
                - Main gaps:
                - Comparison with ground truth:
                [End Evaluation]
                [Improvement Suggestions]
                1.
                2.
                3.
                [End Suggestions]
                Based on your above analysis, provide numerical scores in the following format:
                [Scores]
                Relevance: <score>/5
                Coherence: <score>/5
                Academic: <score>/5
                Completeness: <score>/5
                Innovation: <score>/5
                Total: <sum>/25
                [End Scores]
                Below are the materials for evaluation:
                Paper Title:
                {item["title"]}
                Abstract:
                {item["abstract"]}
                Ground Truth Content:
                {gold_generation}
                AI Generated Content:
                {gen}
                Remember to first provide detailed evaluation, then improvement suggestions, and
                finally the numerical scores in the exact format specified above.
            """

            # (judge both at once by contrasting them?)
            judge_responses = []
            for gen in [sc_generated_paper, sc_pr_generated_paper]:
                chat = judge_tokenizer.apply_chat_template(
                    [{"role": "user", "content": judge_instruction(gen)}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                judge_input = judge_tokenizer([chat], return_tensors="pt", padding=True, padding_side="left").to(config["judge_device"])
                res = judge.generate(
                    **judge_input,
                    max_new_tokens=generation_breakpoint,
                    temperature=0.6
                )
                res = judge_tokenizer.batch_decode(res)
                judge_responses.append(res)
                print(f"judging{len(judge_responses)} done")
            
            samples += 1
            if samples >= max_samples:
                break

        except ScholarCopilotRetrievalError: 
            retrieval_fails += 1 # skip this sample entirely

        

    save_results({
        "eval_dataset": eval_dataset_path,
        "retrieval_index": index_dir,
        "num_samples": samples,
        "shuffled_samples": shuffle,
        "generation_breakpoint": generation_breakpoint,
        "SC retrieval fails": retrieval_fails,
        "SC generated paper": sc_generated_paper,
        "SC PR generated paper": sc_pr_generated_paper,
        "SC judge response": judge_responses[0],
        "SC PR judge response": judge_responses[1]
    }, config, mode="eval_generation")
