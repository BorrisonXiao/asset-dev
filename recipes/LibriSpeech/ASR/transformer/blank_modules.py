"""Small learnable pieces for the blank/separator-token ablations.

Kept separate so the yaml can `!new:blank_modules.LearnedVector` it and the
Checkpointer/optimizer pick it up like any other module.
"""

import torch
import torch.nn as nn


class LearnedVector(nn.Module):
    """A single learned vector — the ``blank_mode: const`` replacement.

    Replaces every separator ("blank") segment embedding with one shared,
    trained vector (applied pre-projection, so all separator tokens become
    identical after ``proj``). This is the content-free "register/delimiter"
    token test: if it recovers WER, the separator *content* was not what
    mattered (H2/H4); if not, the content did (H5).

    Arguments
    ---------
    dim : int
        Feature dimension (``ssl_feat_dims``; applied before ``proj``).
    init_std : float
        Std of the zero-mean Gaussian initialization.
    """

    def __init__(self, dim, init_std=0.02):
        super().__init__()
        self.vec = nn.Parameter(torch.randn(dim) * init_std)

    def forward(self):
        """Return the learned vector ``(dim,)``."""
        return self.vec
