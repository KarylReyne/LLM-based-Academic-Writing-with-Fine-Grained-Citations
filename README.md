# LLM-based Academic Writing with Fine-Grained Citations
## Included Evaluation Data
The evaluation results reported in the thesis are included in `src/out_thesis/`. New evaluation results will also be recorded in this folder.
## How to reproduce the results from the thesis
Start by installing the dependencies: 
```
pip install -r requirements.txt
```
Next, download the required models and some of the datasets:
```
cd src
sh download_models.py
```
Note that you will additionally need the HDT dataset `documents_3.0.json`. Place this dataset in the `src/data/` folder.
To augment this dataset with the arxiv ids for (most of) the cited documents, run
```
python dataset_obtain_arxiv_ids.py
```
Create a smaller version of the dataset (to make future dataset creation faster) by running
```
python dataset_process_corpus.py
```
The next processing steps require, depending on the configuration set in the corresponding `config_*` file, up to 4 GPU devices. Remember to allocate your resources accordingly. To perform the single-step retrieval evaluation, run
```
python evaluation_retriever_thesis.py
```
For the generation quality evaluation run
```
python evaluation_generation.py
```
Note that these files create a lot of different `json` files the first time they are executed which may take a while, please plan accordingly. Executions after that will be a lot faster once all required files are generated.
If you're interested how the plots in the thesis were generated, have a look at `plots_thesis.py`.
## Configuration Guide (for `cfg/config_*.json`)
 - `retriever`: The **additional** retriever used by the PR module, see section 4.4 in the thesis. Supported is `"reasonir_8b"`.
 - `reranker`: The passage retriever used by the PR module, see section 3.4 and 4.1 in the thesis. Supported is `"qwen2.5_7b_instruct"`.
 - `judge`: The model used for generation quality evaluation, see section 4.1 in the thesis. Supported is `"deepseek-r1-distill-qwen-7b"`.
 - `*_device`: GPU on which the corresponding model will be executed.
 - `m_reranking`: Number of LLM calls used for Batched Self-Consistency (PR module). Denoted as `m_PR` in the thesis.
 - `m_judging`: Number of LLM calls used for Batched Self-Consistency (judge). Denoted as `m_judge` in the thesis.
 - `final_score_delta`: Parameter for combining two passage retriever scores, see section 4.4, Eq. 4.1 in the thesis where it is denoted as `\delta`.
 - `ranking_score_normalization`: Whether passage ranking scores should be normalized using min-max normalization.
 - `passage_length`: Size of the passages used by the PR module. Counted in tokens as determined by the ScholarCopilot tokenizer.
 - `label_sep_token`, `tokenizer_begin_token`, `citation_mask_token`, `special_tokens`, `query_expansion`: No **not** change these.
 - `enable_query_context_window`, `query_context`: Whether to use a context window for the PR module and what size (in tokens) it should be respectively. The size is denoted as `w` in the thesis.
 - `sc_retriever_topk`: Number of documents passed from ScholarCopilot to the PR module. Denoted as `k_SC` in the thesis.
 - `enable_passage_retriever`: Whether to use an **additional** retriever (ReasonIR) for the PR module.
 - `retriever_topk`: Number of passages passed from the **additional** retriever to the normal passage retriever. Denoted as `k_ReasonIR` in the thesis.
 - `use_only_passage_retriever`, `only_passage_retriever_topk`: Used for a cut feature. Should always be **false** and a positive integer respectively.
 - `reranker_topk`: Number of passages returned by the passage retriever. Should be >= 1. This is effectively only useful for debugging since only the first passage is used for generation.
 - `retr_batch_size`: Batch size used for the **additional** retriever (ReasonIR).
 - `rera_sc_permutation_mode`: Permutation strategy used for Batched Self-Consistency. Supported are `"stb"` and `"bts"`. See [Korikov et. al. 2025](https://arxiv.org/abs/2505.12570) for details. Denoted as `psg. permutation` in the thesis (section A.1).
 - `rera_passages_per_call`: Number of passages per batch used for Batched Self-Consistency. See [Korikov et. al. 2025](https://arxiv.org/abs/2505.12570) for details. Denoted as `psg. per batch` in the thesis (section A.1).
 - `reranker_temperature`, `judge_temperature`: Passage retriever / judge Softmax temperature respectively.
 - `hnsw_efSearch`: `efSearch` parameter used for ScholarCopilots' HNSW index retrieval.
 - `judge_should_only_score`: Whether the judge should only generate scores (**true**) or reasoning + scores (**false**). See section 4.1 in the thesis.
 - `RECALL_K`: Recall `k` used for evaluation. May override `*_topk` settings, see Table A.1 in the thesis. If you don't want your `*_topk` settings overridden, modify `evaluation_retriever_thesis.py`, line 83-97 accordingly.
 - `SECTIONS`: Which dataset to use for retrieval evaluation. Supported are `"intro+relwork"`, `"methods"`, `"experiments"` and `"conclusion"`. See section 4.1 in the thesis.
 - `with_abstracts`: How citations should be replaced in citing contexts during retrieval evaluation. Setting this to **false** is equivalent to the `masked` strategy described in the thesis (see section 4.1) 
 - `shuffle_samples`: Whether evaluation samples should be shuffled. Works for both retrieval and generation evaluation.
 - `num_samples`: Number of samples to evaluate. Works for both retrieval and generation evaluation.
 - `max_samples`: Maximum number of evaluation samples to load into memory. Should be larger that `num_samples`.
 - `generation_breakpoint`: Maximum number of tokens to generate (by the evaluated models) during generation evaluation.
 - `shuffle_instruction`: Whether the generation quality dimensions (see section 4.1 in the thesis) should be shuffled before the judge instruction is put together.
## License
This software is released under a BSD 3-Clause License. A copy of this license can be found in LICENSE.txt. It makes use of and is distributed with substantial portions of modified code from [ScholarCopilot](https://github.com/TIGER-AI-Lab/ScholarCopilot/tree/main) which is licensed under the MIT license. A copy of their license can be found in license/ScholarCopilot.LICENSE. Source files derived from ScholarCopilot code are marked with the prefix `scholarcopilot_`.