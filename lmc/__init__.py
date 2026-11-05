"""Code for two studies of linear mode connectivity in small MLPs.

lmc.mnist  -- symmetry-breaking architectures and midpoint-trained pairs on MNIST
lmc.fills  -- how the frozen fill values of a W-Asymmetric mask affect the barrier
"""

import hashlib


def stable_seed(*parts) -> int:
    """Integer seed derived from md5 of the arguments.

    Unlike Python's built-in hash(), this is identical across processes, so
    buffers seeded with it (masks, fills, gates) match between runs.
    """
    digest = hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()
    return int(digest, 16) % (2**31)
