### Modules
 - Retriever
     - given a query x, retrieves K relevant documents (or chunks ?) from a data collection (scientific papers, e.g. arxiv)
     - ReasonIR
 - Reranker
     - given a query x and K documents, return k re-ranked documents
     - ReasonIR-Rerank
 - Generator
     - given a query x and k documents, generate an initial response y0=s1C1...snCn which is composed of statements si and their corresponding set of citations Ci=ci1...cim where each citations cij supports si
     - use self-feedback to update response to yk
     - OS-8B (OpenScholar)
 - PassageExtractionModule
     - condense document citations cij to fine-grained sentence citations c'ij based on the corresponding generated statement si (maybe retrieval with entailment analysis or splitting/chunking + ranking with ContextCite)
     - idea taken from LongCite's CoF method

### Dataset Generation
 - requirements by module
     - retriever
         - match query q to documents that contain information about q
         - documents should be scientific to maximize the reliability and accuracy of the information it contains
     - reranker
         - same as retriever
     - generator
         - given q and D, generate statements s based on d in D that address q
         - generated statements should mimic scientific writing, sources for citations are directly taken from D
     - passage extractor
         - given s and d (where d already supports s), identify the span p (sentence?) that informs s the most. idealy, identify the span that caused the model to generate s in the first place
         - p should be as semantically similar to s as possible, to approximate human citations well
 - possible datasets to draw from
     - general retrieval: MS MARCO, Natural Questions, HotpotQA, [arxiv corpus](https://www.kaggle.com/datasets/Cornell-University/arxiv) 
     - longform QA: ASQA, ELI5
     - scientific claim-document mapping: SciFact
     - scientific QA: KIWI
 - methods for dataset generation that could be useful
     - ReasonIR-Synthesizer (generation of synthetic data)
         - VL: prompt LLAMA3.1-70B-INSTRUCT to output a long (300-2k) query and corresponding positive document (p.19, fig. 8)
         - HQ: prompt LLAMA3.1-70B-INSTRUCT to formulate a difficult question given a seed document (p.17, fig. 4). The LLM has to reason about its generation. The seed document is used as the corresponding positive document
         - useful for the **retriever/reranker**
     - [Science Hierarchography](https://arxiv.org/abs/2504.13834) (synthetic contribution extraction)
         - prompt GPT-4o to extract structured contributions from given papers (p. 17, fig. 3)
         - maybe useful for generating query-document pairs for the **retriever/reranker**
     - [LongAlign](https://arxiv.org/abs/2401.18058) (generation of synthetic training data)
         - generate questions and answers according to a given long context (p.3, fig. 2)
         - useful for the **generator**
     - [SciFact](https://aclanthology.org/2020.emnlp-main.609/)
         -  expert-written scientific claims in the biomedical domain, paired with gold evidence from existing PubMed paper abstracts 
         - useful for the **generator**
     - [KIWI](https://arxiv.org/abs/2403.03866) (**human** question refinement instructions)
         - a researcher iteratively issues instructions for a LLM to revise its answer to a given question
         - useful for the **generator** and maybe an additional answer refinement module
         - if combining multiple turns of refinement instructions into one string, that may be used as a longform question (to enrich ReasonIRs' HQ?)
     - the **PassageExtractionModule** might not need its own dataset if it is based on semantic similarity...?

_look for more human-annotated data!!!_
 
### Misc
 - [PaperQA](https://arxiv.org/abs/2312.07559)
 - [PaperQA2](https://arxiv.org/abs/2409.13740)
 - [OpenResearcher](https://arxiv.org/abs/2408.06941) for retrieval refinement methods