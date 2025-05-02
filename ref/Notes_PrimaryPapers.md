## PAPERS
### ContextCite https://arxiv.org/abs/2409.00729
 - context attribution (CA): Can we pinpoint the parts of the context (if any) that led to a particular generated statement?
 - CA is done by an independent linear 'surrogate model' tau -> backbone agnostic
 - tau accepts a list of d sources s1...sd and assigns a score to each source indicating its 'importance' to the response
 - CA implements _contributive attribution_: identify the sources that cause a model to generate a statement (_corroborative attribution_ would identify sources that support or imply a statement)
 - Top-k log-probability drop
    - log-prob(tau)-log-prob(tau_{top-k sources removed})
    - metric for comparing methods for context attribution
 - CA works as follows
    - Sample a 'training dataset' of ablation vectors v1...vn and compute R=LM(vi) for each
    - Learn a surrogate model tau that approximates LM by training on the pairs (vi,LM(vi))
    - Attribute the behavior of tau to the sources removed in each vi
 - CA eval on: TyDi QA, Hotpot QA, CNN DailyMail
 - backbone: instruction-tuned Llama-3-8B, Phi-3-mini

### REASONS https://arxiv.org/abs/2405.02228
 - dataset/benchmark
 - metadata-augmentation reduces hallucinations
 - progressively adding metadata (abstract followed by author information) enables successful identification of the correct citation
 - 2017-2024
 - contains extracted from related work sections of IEEE-formatted papers in Computer Science and Biology published on ArXiv
 - dataset creation:
    - isolate related work section (parsing)
    - extract individual sentences containing citations
    - sentences are stored with json metadata, including: source paper ID and pub. date, citation target metadata, cross-domain citation markers, surrounding context sentences, classification tag
    - only CC Zero, CC BY and CC BY-SA
 - benchmark tasks
    - Direct Query: LLM generates author names when given a paper title (author attribution)
    - Indirect Query: LLM generates the cited paper title when given a sentence (title attribution)
    - SID Prompting: LLM tasked to identify citations based solely on the quoted text, if LLM is uncertain, (incrementally ?) provide additional information such as complete author information, full abstract text, and additional contextual signals
 - RAG architecture
    - given sentence s and retrieval corpus R
    - encode s with query encoder, d in R with document encoder
    - retrieve top-k relevant documents for s based on embedding similarity (OpenAI bi-encoder)
    - similarity measure: BM25 or MPNet cross-encoder trained on REASONS doc. index with contrastive loss
 - proposed eval metrics
    - Hallucination Rate quantifies the model’s tendency to generate incorrect or partially correct citations
    - Pass Percentage measures the model’s discretion in responding, showing its ability to abstain when uncertain

### HAtten https://arxiv.org/abs/2112.01206
 - addresses local citation recommendation as a retrieval task, a query consists of two 'contexts':
    - text surrounding the citation placeholder ('local context')
    - title and abstract of the citing paper ('global context')
 - recommender pipeline:
    - prefetching
       - embed query and documents with bi-encoder
       - retrieve k nearest neighbors of query (using knn, so probably just vector space distance)
       - encode query and papers with a two-layer Hierarchical Attention-based text encoder (HAtten) (paragraph embedding -> query/doc embedding)
    - re-ranking
       - SciBERT
 - dataset
    - 3.2m local citation sentences (with title+abstract of both the cited and citing paper)
    - they also experiment on: ACL-200, RefSeer, FullTextPeerRead

### TODO: Generation Model

### TODO: Alternate Retriever: [ReasonsIR](https://arxiv.org/pdf/2504.20595)
 - TODO
