
with open('./siibra/VERSION', 'rt') as fp:
    version = fp.read().strip()
print("Version:", version)
with open('./CITATION.cff', 'rt') as fp:
    matches = [line for line in fp.readlines() if line.startswith('version')]
    assert len(matches) == 1, "There should be only one line for version in CITATION.cff"
    version_in_citation = matches[0].removeprefix('version: v').strip()
print("Version in CITATION.cff:", version_in_citation)

assert version == version_in_citation, f"The version in CITATION.cff ('{version_in_citation}') and VERSION ('{version}') file does not match."
print('Versions match.')
