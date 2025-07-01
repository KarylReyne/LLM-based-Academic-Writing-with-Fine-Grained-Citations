

from latex_parsing import download_from_arxiv, TexParsingError
from passage_retrieval_interface import get_candidate_passages, LABEL_SEPARATOR


def download_paper_texfiles(paper_ids):
    downloaded_ids = []
    for id in paper_ids:
        try:
            download_from_arxiv(id)
            downloaded_ids.append(id)
        except TexParsingError:
            continue
    return downloaded_ids

def get_initial_generation_context(paper_id, dataset, passage_retrieval_models, config):
    NUM_INTRO_TOKENS = 32

    title = dataset[paper_id]["title"]
    abstract = dataset[paper_id]["abstract"].lstrip("<|reference_start|>").rstrip("<|reference_end|>")

    tokenizer = passage_retrieval_models["retr_tokenizer"]
    sections = get_candidate_passages(paper_id, tokenizer, config)
    introduction = ""
    for s in sections:
        label, text = s.split(LABEL_SEPARATOR)
        label = label.split("_")[1]
        if not label.startswith("abstract"): # should be the first section after the abstract
            tokens = tokenizer(text).to(config["retriever_device"])
            text = tokenizer.decode(tokens[:NUM_INTRO_TOKENS])
            introduction = text
            break

    return f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction:\n{introduction}"


if __name__ == "__main__":
    
