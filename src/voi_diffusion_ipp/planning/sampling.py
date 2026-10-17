"""Optional variance reduction for Gaussian outer predictive worlds.

Pairing changes joint sampling, not any world's marginal distribution. RPCBF's
independent inner draws are never paired with outer worlds or each other.
"""


def couple_outer_samples(eps, noise, worlds, mode="iid"):
    if mode == "iid":
        return eps, noise
    if mode != "antithetic":
        raise ValueError(f"Unknown outer sampling mode: {mode}")
    if worlds < 2 or worlds % 2:
        raise ValueError("Antithetic outer sampling requires a positive even world count")
    if len(eps) < worlds or len(noise) != worlds:
        raise ValueError("Outer field and observation-noise dimensions disagree")
    # Use independent base draws for each pair. The remaining rows of eps are
    # exactly the original iid inner fields used by the execution/safety filter.
    eps = eps.at[1:worlds:2].set(-eps[:worlds:2])
    noise = noise.at[1:worlds:2].set(-noise[:worlds:2])
    return eps, noise
