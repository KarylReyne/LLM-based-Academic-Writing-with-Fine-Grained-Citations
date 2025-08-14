

from dataset_loaders import load_generation_eval_dataset, arxiv_to_corpus_id, scholarcopilot_arxiv_to_corpus_id, load_retrieval_dataset

if __name__ == "__main__":
    # load deepseek-r1

    # build/load dataset of left-side generation contexts (just take title+abstract+2.5 sentences introduction of existing papers)
    docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
    processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
    docs_corpus_id_map = arxiv_to_corpus_id(docs_id_map_path, processed_corpus_path)
    docs_arxiv_id_map = {v: k for k, v in docs_corpus_id_map.items()}

    sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
    sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
    sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)
    sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
    
    docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
    complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
    docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

    eval_dataset_path = "data/eval_dataset_generation.jsonl"
    eval_dataset, eval_indices = load_generation_eval_dataset(eval_dataset_path, docs_retrieval_dataset, sc_arxiv_id_map, max_samples=1000, shuffle=True)
    # generate with SC
    # generate with SC+PR
    # judge each generated output individually (prompt from SC paper)
    # (judge both at once by contrasting them?)

    print(eval_dataset[:5])