from siibra.retrieval_new.api_fetcher import brainglobe
import siibra

for filename in brainglobe.ls():
    print(filename)

space, parc = brainglobe.use(brainglobe.vocab.ALLEN_MOUSE_100UM_V1_2)


got_parc = siibra.get_parcellation(f"bg-{brainglobe.vocab.ALLEN_MOUSE_100UM_V1_2}")
got_space = siibra.get_space(f"bg-{brainglobe.vocab.ALLEN_MOUSE_100UM_V1_2}")

assert parc is got_parc
assert space is got_space
