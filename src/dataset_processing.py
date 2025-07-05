import json
import requests
import urllib, urllib.request


def get_arxiv_id_from_citation(citation):
    arxiv_id = None
    id_type = None
    if len(citation.split(" abs/")) == 2:
        arxiv_id = citation.split(" abs/")[1].split(",")[0].split(" ")[0].rstrip(".")
        id_type = "extracted"
    else: # arxiv title search
        ARXIV_MAX_RESULTS = 10
        title = ""
        for substring in citation.split(".")[1:]:
            if substring.startswith("  "): # titles seem to always be preceded by "  "
                title = substring.lstrip("  ")
                break
        title_for_url = title.replace(" ", "+").replace(":", "%3A").replace("\"", "%3F")
        url = f'https://export.arxiv.org/api/query?search_query={title_for_url}&searchtype=title&start=0&max_results={ARXIV_MAX_RESULTS}'
        try: 
            data = urllib.request.urlopen(url) # sometimes returns 400
            xml = data.read().decode()
            xml = xml.split("<entry>")[1:] # skip to the actual search results
        except urllib.error.HTTPError as e:
            print(title)
            print(url)
            raise e
        except IndexError:
            # no search results returned
            raise ArxivSearchFailedError
        for entry in xml:
            entry = entry.replace(">\n", ">SPLIT").split("SPLIT")
            entry_title = ""
            entry_id = ""
            for entry_item in entry:
                entry_item = entry_item.lstrip().replace("\n", " ")
                if entry_item.startswith("<title>"):
                    entry_title = entry_item.lstrip("<title>").rstrip("</title>")
                elif entry_item.startswith("<id>"):
                    item_body = entry_item.lstrip("<id>").rstrip("</id>")
                    entry_id = item_body.split("/")[-1]
            # check if the entry is about the correct paper by comparing titles
            process_title = lambda t: t.lower().replace(" ", "")
            t1 = process_title(title) # citation
            t2 = process_title(entry_title) # search result entry
            if not t1 == t2:
                # print(f"t1 {t1}")
                # print(f"t2 {t2}")
                continue # try the next search result
            else:
                arxiv_id = entry_id
                id_type = "searched"
                break # found the paper
        if arxiv_id == None:
            # print(citation)
            # print(title)
            raise NoResultMatchedError
        
    assert arxiv_id != None, citation
    # arxiv_id = arxiv_id.split("v")[0] # remove version
    return arxiv_id, id_type


class ArxivSearchFailedError(Exception):
    """The arxiv title search did not yield the correct result"""
    pass

class NoResultMatchedError(Exception):
    """The arxiv title search did not yield the correct result"""
    pass


if __name__ == "__main__":
    dataset = {}
    dataset_path = "data/documents_3.0.json"

    DEBUG_LEN = 200
    with open(dataset_path, "r") as file:
        for line in file:
            rec = json.loads(line)
            dataset[rec["arxiv_id"]] = rec
            if len(dataset) >= DEBUG_LEN:
                break
    
    ids = []
    ids_in_dataset = 0
    num_extracted_ids = 0
    num_searched_ids = 0
    search_fails = 0
    match_fails = 0
    for arxiv_id in dataset:
        bib = dataset[arxiv_id]["bibliography"]
        for k in bib:
            try:
                id, id_type = get_arxiv_id_from_citation(bib[k])
                ids.append(id)
                try:
                    _ = dataset[id]
                    ids_in_dataset += 1
                except KeyError:
                    pass
                if id_type == "extracted":
                    num_extracted_ids += 1
                elif id_type == "searched":
                    num_searched_ids += 1
            except ArxivSearchFailedError:
                search_fails += 1
            except NoResultMatchedError:
                match_fails += 1
    print(ids)
    print(f"ids: {len(ids)}")
    print(f"ids_in_dataset: {ids_in_dataset}")
    print(f"num_extracted_ids: {num_extracted_ids}")
    print(f"num_searched_ids: {num_searched_ids}")
    print(f"search_fails: {search_fails}")
    print(f"match_fails: {match_fails}")
