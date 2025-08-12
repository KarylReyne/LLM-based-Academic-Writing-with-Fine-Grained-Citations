import faiss
import os
import h5py
import numpy as np


def load_corpus_base(file_path):
    encoded_list = []
    lookup_indices_list = []

    print(f"Loading embedded vectors...")
    try:
        with h5py.File(file_path, 'r') as f:
            encoded_list.append(f['encoded'][:])
            lookup_indices_list.append(f['lookup_indices'][:])
        print(f"Successfully loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

    if encoded_list and lookup_indices_list:
        encoded = np.concatenate(encoded_list, axis=0)
        lookup_indices = np.concatenate(lookup_indices_list, axis=0)
        print(f"Combined shape - encoded: {encoded.shape}, lookup_indices: {lookup_indices.shape}")
        print("embedded vectors loaded.")
        return encoded, lookup_indices
    else:
        raise ValueError("No data was successfully loaded")


if __name__ == "__main__":
    embed_dim_map = {
        "scholarcopilot": 3584,
        "reasonir_8b": None #TODO
    }

    input_file = "data/documents_3.0_512-passages_reasonir_8b-encoded_corpus.h5"
    threads = 1
    embed_dim = embed_dim_map["scholarcopilot"] # tied to the encoded corpus
    M = 16 # number of edges that need to be added to every new node during insertion
    efC = 1000 # number of nearest neighbors to explore during construction

    faiss.omp_set_num_threads(threads)

    vectors, lookup_indices = load_corpus_base(input_file)

    print('loaded vectors')
    index = faiss.IndexHNSWFlat(embed_dim, M, faiss.METRIC_INNER_PRODUCT)
    print('created index')
    index.hnsw.efConstruction = efC
    index.verbose = True

    print('adding vectors')
    faiss.normalize_L2(vectors)
    index.add(vectors)
    print('vectors added')

    faiss.write_index(index, input_file.replace("_corpus.h5", '_index'))

    with open(input_file.replace("_corpus.h5", '_lookup_indices.npy'), 'wb') as f:
        np.save(f, lookup_indices)



