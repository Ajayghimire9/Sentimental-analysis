from src.drift import population_stability_index


def test_psi_is_zero_for_identical_population():
    values = list(range(1, 101))
    assert population_stability_index(values, values) == 0.0
