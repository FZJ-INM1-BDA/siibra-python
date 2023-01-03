
def set_test_cache():
    from siibra import _retrieval
    _retrieval.CACHEDIR = retrieval.__compile_cachedir(suffix='pytest')
    print('Using test cache: {}'.format(_retrieval.CACHEDIR))
    _retrieval.clear_cache()
