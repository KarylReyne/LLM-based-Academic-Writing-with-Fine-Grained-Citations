import torch
import json
from transformers import AutoModel, AutoTokenizer
from torch import nn
from datetime import datetime

from latex_parsing import get_citations_data_from_bbl, get_source_citations


MODEL_IDENTIFIER = "reasonir/ReasonIR-8B"
LABEL_SEPARATOR = "<LABEL-SEP>"


# srun --job-name "ReasonIRtest" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash
# cd src
# conda activate citations
if __name__ == "__main__":

    # for id in ["1607.06450"]:# , "1409.0473", "1703.03906"]:
    #     # dl_arxiv(id)
    #     path = get_bbl_path_from_arxiv_id(id)
    #     citations_data, ids = get_citations_data_from_bbl(path)
    #     # print(citations_data)

    #     # recursion_depth = 3
    #     # ids_to_process = ids
    #     # new_ids = []
    #     # while recursion_depth > 0:
    #     #     for x in ids_to_process:
    #     #         try:
    #     #             dl_arxiv(x)
    #     #         except tarfile.ReadError:
    #     #             continue # skip if extraction failed

    #     #         _path = get_bbl_path_from_arxiv_id(x)
    #     #         _, _new_ids = get_citations_data_from_bbl(_path)
    #     #         [new_ids.append(y) for y in _new_ids]

    #     #     ids_to_process = new_ids
    #     #     new_ids = []
    #     #     recursion_depth -= 1


    id = "2108.09084" # Fastformer
    citations_data, _ = get_citations_data_from_bbl(id, additional_citation_records={
        "vaswani2017attention": {
            "bib_id": "vaswani2017attention",
            "arxiv_id": "1706.03762", 
            "title": "Attention is all you need", 
            "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.", 
            "name": "Ashish Vaswani Noam Shazeer Niki Parmar Jakob Uszkoreit Llion Jones Aidan~N Gomez Lukasz Kaiser Illia Polosukhin"
        }
    })
    # print(len(citations_data.items()))

    target_bib_id = "vaswani2017attention" # Transformer
    target_citation_record = citations_data[target_bib_id]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)

    citing_sents, target_doc_sections = get_source_citations(id, target_citation_record, tokenizer)

    model = AutoModel.from_pretrained(MODEL_IDENTIFIER, torch_dtype="auto", trust_remote_code=True)
    model = model.to("cuda")
    model.eval()

    citing_sent = citing_sents[0]
    similarity_records = {}
    for candidate_chunk in target_doc_sections:
        query_instruction = "" 
        doc_instruction = ""

        # TODOs
        # experiment with instructions, specify the mask token
        # test query context length
        # replace figure with captions text

        # https://arxiv.org/abs/2505.12570

        t = candidate_chunk.split(LABEL_SEPARATOR)
        chunk_label = t[0]
        chunk = t[1]

        query_emb = model.encode(citing_sent, instruction=query_instruction)
        doc_emb = model.encode(chunk, instruction=doc_instruction)
        sim = query_emb @ doc_emb.T

        similarity_records[chunk_label] = {
            "query": citing_sent,
            "chunk": chunk,
            "sim": f"{sim}"
        }

    similarity_records = dict(sorted(similarity_records.items(), key=lambda item: item[1]["sim"], reverse=True))

    with open('out/similarity_records.json', 'w', encoding='utf-8') as f:
        json.dump({
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}": {
                "query_doc_id": f"{id}",
                "target_doc_id": f"{target_citation_record["arxiv_id"]}",
                "records": similarity_records
            }
        }, f, ensure_ascii=False, indent=4)