
def set_test_cache():
    from siibra import retrieval
    retrieval.CACHEDIR = retrieval.__compile_cachedir(suffix='pytest')
    print('Using test cache: {}'.format(retrieval.CACHEDIR))
    retrieval.clear_cache()
