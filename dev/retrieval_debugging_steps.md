 - are similar words actually closely embedded?
     - fine-tuning
     - same query/index preprocessing?
     - correctly normalized?
 - index stored correctly?
     - retrieve and check indexed vectors, do they all have the same dim?
     - similarity during indexing is the same as for retrieval?
     - test different index.nprobe (higher => higher recall but slower) - was 
     - test different ef_search (higher => higher recall but slower) - was (efC was 1000)
     - test different M (higher => higher accuracy but more memory) - was 16 / optim_index 32
     - efC - was 1000 / optim_index 1500

reasonir skript: https://github.com/facebookresearch/ReasonIR/blob/main/evaluation/bright/retrievers.py#L711
