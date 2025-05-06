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

### Generator: [OpenScholar](https://arxiv.org/pdf/2411.14199)
 - relevant passage retrieval with refinement through self-feetback generation
 - RAG with retrieve-then-generate
 - ScholarQABench: multi-domain benchmark for literature search, 2.9k expert-written queries, 208 answers
 - OpenScholar-DataStore: passage retrieval corpus ([peS2o v3](https://aclanthology.org/2024.acl-long.840/), ranges until Oct. 2024), 45m papers from SemanticScholar with 237m corresp. passage embeddings
 - OpenScholar inference pipeline:
    - given query, retrieve relevant papers (from peS2o, abstracts from papers returned via the Semantic Scholar API and papers retrieved by a web search engine using the query)
    - retrieve (bi-encoder) candidate top N relevant papers from OSDS
    - rerank (cross-encoder) candidate papers
    - LM synthesizes the retrieved passages to generate an initial response y0
    - given y0, the LM then generates self-feedback f1 (natural language sentence that describes potential improvements)
    - feedback is used to iteratively refine/update the response to yk
 - retriever: [Contriever](https://openreview.net/forum?id=jKN1pXi7b0) pre-trained unsupervised on peS2o
 - reranker: [BERT-based reranker](https://arxiv.org/abs/1901.04085), specifically [BGE-reranker](https://arxiv.org/abs/2309.07597) fine-tuned on synthetic data generated by Llama3-70B-Instruct

### Retriever: [ReasonsIR](https://arxiv.org/pdf/2504.20595)
 - reasoning-intensive document retrieval 
 - ReasonIR-Synthesizer: synthetically generating two types of reasoning-intensive retrieval data
    - _varied-length queries_ (VL) and their corresponding synthesized documents, which are of diverse lengths and are designed to extend the effective context length
       - task LLM to generate a positive document for a given query
    - _hard queries_ (HQ), reasoning-intensive queries generated based on real seed documents
       - [BRIGHT](https://arxiv.org/abs/2407.12883) as the initial document pool, scored by the FineWeb-Edu classifier for educational value
       - task LLM to reason about the background knowledge, common problem-solving patterns, and realistic scenarios before formulating a difficult question while also avoiding seed-document-dependent (e.g. using terms specific to the seed doc) questions
       - seed doc serves as positive document
    - each includes synthetic hard negatives, documents that appear superficially relevant but are actually unhelpful for the query
       - generate the hard negative in a separate turn, conditioning on the previously obtained query and positive document for both VL and HQ data
 - ReasonIR-8B bi-encoder
    - LLAMA 3.1-8B with a bi-directional attention mask, fine-tuned on public datasets and synthetic data (ReasonIR-Synthesizer)
    - consistently benefits from longer rewritten queries
       - rewriting seeks to make the content of the query more lexically and semantically relevant (such that the retiever can retrieve more relevant top-k documents)
       - Reason-query: query rewriter g(·,c) with a length configuration c and chain-of-thought reasoning, producing a rewritten query q̃ = g(q,c)
    - benefits from additional LLM reranking
       - ReasonIR-Rerank: interpolate the reranking scores with the scores given by the base retriever
       - QWEN2.5-32B-INSTRUCT
    - contrastive loss: optimizes the retriever h to embed queries q closer to relevant documents D+ than to irrelevant ones D−
    - computing distances with all D− is expensive -> train retriever only on hard (difficult) negative documents d~ in D− for which cos sim to q is large
 - train data
    - 1.38m public (MS MARCO, Natural Questions, HotpotQA)
    - 245k VL
    - 100k HQ
 - eval datasets
    - IR: BRIGHT
    - RAG: MMLU, GPQA