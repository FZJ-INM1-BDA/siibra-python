import os

from trx.fetcher import fetch_data, get_home, get_testing_files_dict

from siibra.retrieval.requests import FileLoader


def test_loading_trx():
    # Download test data
    fetch_data(get_testing_files_dict(), keys="gold_standard.zip")
    trx_home = get_home()
    trx_path = os.path.join(trx_home, "gold_standard", "gs.trx")

    # create request
    req = FileLoader(trx_path)

    # load TRX file
    req.get()
