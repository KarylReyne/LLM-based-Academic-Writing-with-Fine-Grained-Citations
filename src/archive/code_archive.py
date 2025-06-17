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