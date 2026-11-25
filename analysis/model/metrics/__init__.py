"""Expose components.
"""

from . import gram
from . import loss

"""Expose interfaces.
"""

from .gram import gram_euclidean, gram_cosine, get_gram_fn
from .loss import get_loss_fn, loss_geometrical, loss