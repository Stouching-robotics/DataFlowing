"""Self-contained black-glove detection support for the Data Acquisition worker.

The workflow adapter imports this package directly.  It intentionally does not
depend on the legacy repository-level demo directories.
"""

from .contracts import DetectedHand

__all__ = ["DetectedHand"]
