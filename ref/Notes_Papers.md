## PAPERS
### ScholarCopilot
 - generation of academic writing, only introduction and related work
 - RAG variant
 - continuous generation stage, generates retrieval tokens that pause generation to retrieve additional information, usually references (abstracts or key excerpt)
 - enables user refinement via 
 - dataset of 501k arxiv papers and 16.8k citation titles matched to either arxiv or semantic scholar
 - unified loss Lg + lambda*Lr where Lg is max log-likelihood conditioned on prev token and Lr is contrastive loss that differentiates relevant from irrelevant citations
 - generation model:  Qwen-2.5-7B-re, Qwen-2.5-7B-Instruct, Qwen-2.5-72B-re
 - retrieval model: E5-Mistral-7B-Instruct

### LongCite https://arxiv.org/pdf/2409.02897
 - generation with sentence-level citations via in-context learning
 - reduces hallucinations, provides verifiability and faithfulness
 - LongBench-Cite: benchmark for evaluating LLM for long-context question answering with citations (data from LongBench and LongBench-Chat, English and Chinese1)
 - Coarse to Fine (CoF): method for constructing long-context QA instances with citations
    - LLM produces query and answer from long-context input (via [Self-Instruct](https://aclanthology.org/2023.acl-long.754/))
    - use the answer to retrieve chunks (128 tokens) from the context that are fed to the LLM as (coarse) citations within the answer
    - LLM identifies relevant sentences within each chunk which then form (fine-scale) citations
    - instances with an insufficient number of citations are discarded
 - LongCite-45k: dataset that consists of 44.6k "long-context QA with citations" instances with contexts up to 128k tokens (obtained via CoF)
 - LongCite-9B and LongCite-8B: generation models based on GLM-4-9B and Llama3.1-8B respectively

### SelfCite https://arxiv.org/pdf/2502.09604
TODO

### OpenScholar https://arxiv.org/pdf/2411.14199
TODO
