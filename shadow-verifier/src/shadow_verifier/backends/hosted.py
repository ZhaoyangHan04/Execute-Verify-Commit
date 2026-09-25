"""Public providers only; credentials are read from environment variables."""

from dataclasses import asdict

from .dashscope import BackendConfigurationError, DashScopeBackend, DashScopeConfig


def hosted_backend(
    model: str,
    config: DashScopeConfig,
    *,
    role: str | None = None,
    default_factory=DashScopeBackend,
):
    if model == "jev-1.13.0":
        if role != "verifier":
            raise BackendConfigurationError(
                "Jev is an action reviewer, not a producer or outcome judge"
            )
        from .typesafe import TypeSafeBackend, TypeSafeConfig

        return TypeSafeBackend(model, TypeSafeConfig(**asdict(config)))
    return default_factory(model=model, config=config)
