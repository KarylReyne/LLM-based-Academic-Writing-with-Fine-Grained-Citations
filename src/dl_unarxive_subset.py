import requests
import os
import tarfile
from zipfile import ZipFile


if __name__ == "__main__":
    # url = "https://zenodo.org/api/records/7752615/files-archive"
    # response = requests.get(url)

    # # download source archive
    # with open(f"data/unarxive_subset.zip", "wb") as handle:
    #     for data in response.iter_content(chunk_size=1024):
    #         handle.write(data)
    #     handle.close()
    # try:
    #     os.makedirs(f"data/unarxive_subset")
    # except FileExistsError:
    #     pass

    # # extract archive
    # zipfile = ZipFile(f"data/unarxive_subset.zip", "r")
    # zipfile.extractall(f"data/unarxive_subset")

    tar = tarfile.open(f"data/unarxive_subset/unarXive_230324_open_subset.tar.xz")
    tar.extractall(f"data/unarxive_subset/unarXive_230324_open_subset")