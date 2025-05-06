## THESIS OBJECTIVES
### Dataset
 - ambiguous queries (academic writing)
 - multiple interpretations per query and their corresponding papers
 - include cases where other systems (Google Scholar, OpenScholar) fail

### Disambiguation System
 - sample interpretations per query
 - generate clarifying questions (which are optimal according to BED)
 - the answers to the clarifying questions are used to narrow down the search space

## PAPERS
### Active Task Disambiguation with LLMs
 - addresses ambiguous task specification (where information is missing and/or the intended behaviour is not clear)
 - iterative task elicitation: (iterative) interactive dialogue between user and LLM with the goal of extracting additional information from the user during each round.
 - BED: experiment optimal <=> experiment maximizes information gain about an unknown quantity of interest
 - Problem Solving Agent (LLM)
     - given a problem statement, samples candidate solutions
     - generates candidate questions that differentiate candidate solutions
     - presents the candidate question with the highest information gain to the oracle
     - updates the problem statement based on the oracle answer
 - Oracle (User)
     - provides the problem statement
     - produces oracle answer when presented with a candidate question
 - LLM agent: GPT-4o-mini, GPT-3.5-turbo, Llama-3-8B, Llama-3-70B
 - Oracle: another LLM, not specified

 ### ScholarCopilot
 - generation of academic writing, only introduction and related work
 - RAG variant
 - continuous generation stage, generates retrieval tokens that pause generation to retrieve additional information, usually references (abstracts or key excerpt)
 - enables user refinement via 
 - dataset of 501k arxiv papers and 16.8k citation titles matched to either arxiv or semantic scholar
 - unified loss Lg + lambda*Lr where Lg is max log-likelihood conditioned on prev token and Lr is contrastive loss that differentiates relevant from irrelevant citations
 - generation model:  Qwen-2.5-7B-re, Qwen-2.5-7B-Instruct, Qwen-2.5-72B-re
 - retrieval model: E5-Mistral-7B-Instruct

### Tasks for next week
1. Dataset for fine-grained citation, especially how to build this kind of dataset automatically (like the LongCite and selfCite paper I sent).
2. Methods for fine-grained citation. LLM-based or light weight model before LLMs.


 ### Reading List
 - [ ] Vaswani et al. 2017
 - [ ] LLaMA/Qwen
 - [ ] SelfCite https://arxiv.org/pdf/2502.09604
 - [ ] OpenScholar https://arxiv.org/pdf/2411.14199
 - [ ] LongCite https://arxiv.org/pdf/2409.02897
