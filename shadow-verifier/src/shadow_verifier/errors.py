"""Kernel errors that are never exposed as normal agent observations."""


class ShadowError(RuntimeError):
    """Base class for Shadow runtime infrastructure failures."""


class ReviewInfrastructureError(ShadowError):
    """A review could not produce a trustworthy semantic decision."""


class SettlementError(ShadowError):
    """The runtime cannot prove the canonical environment's final state."""


class SessionPoisonedError(ShadowError):
    """A previous uncertain settlement made this Shadow session unusable."""
