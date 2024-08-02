from siibra.retrieval.file_fetcher import GithubRepository

def test_github_fetcher():
    repo = GithubRepository("fzj-inm1-bda", "siibra-python", reftag="main", eager=True)
    file = repo.get("LICENSE")
    print(file)
