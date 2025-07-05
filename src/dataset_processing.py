import json


def search_arxiv_for_id(citation):
    


def get_arxiv_id_from_citation(citation):
    arxiv_id = None
    if len(citation.split(" abs/")) == 2:
        arxiv_id = citation.split(" abs/")[1].split(",")[0]
    elif 
    assert arxiv_id != None, citation
    return arxiv_id



if __name__ == "__main__":
    dataset = []
    dataset_path = "data/documents_3.0.json"

    DEBUG_LEN = 100
    with open(dataset_path, "r") as file:
        for line in file:
            rec = json.loads(line)
            dataset.append(rec)
            if len(dataset) >= DEBUG_LEN:
                break
    
    ids = []
    for d in dataset: 
        for k in d["bibliography"]:
            try:
                ids.append(get_arxiv_id_from_citation(d["bibliography"][k]))
            except AssertionError as e:
                print(ids)
                print(len(ids))
                raise e