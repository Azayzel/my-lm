"""Sample buggy module for agent benchmark fixture."""


def divide(a: float, b: float) -> float:
    """Return a / b."""
    return a / b  # BUG: no zero-division guard


def parse_age(raw: str) -> int:
    """Parse a numeric age string and return an int."""
    return int(raw)  # BUG: no error handling for non-numeric input


def greet(name: str | None = None) -> str:
    """Return a greeting."""
    return "Hello, " + name  # BUG: crashes when name is None
