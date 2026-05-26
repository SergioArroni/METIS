"""Registry for report generators."""

from ...domain.contracts import Reporter
from ...domain.errors import RegistryError
from .json_reporter import JSONReporter
from .markdown_reporter import MarkdownReporter


class ReporterRegistryImpl:
    """Concrete implementation of reporter registry."""

    def __init__(self):
        self._reporters: dict[str, type[Reporter]] = {}
        self._register_defaults()

    def register(self, format_name: str, reporter_class: type[Reporter]) -> None:
        """Register a reporter implementation."""
        if not format_name:
            raise RegistryError("Reporter format name cannot be empty", "reporter")

        self._reporters[format_name] = reporter_class

    def get(self, format_name: str) -> type[Reporter]:
        """Retrieve reporter class by format name."""
        if format_name not in self._reporters:
            available = list(self._reporters.keys())
            raise RegistryError(
                f"Reporter not found: {format_name}. Available: {available}",
                "reporter",
                format_name,
            )
        return self._reporters[format_name]

    def list_formats(self) -> list[str]:
        """list available report formats."""
        return list(self._reporters.keys())

    def _register_defaults(self) -> None:
        """Register default reporters."""
        self.register("json", JSONReporter)
        self.register("md", MarkdownReporter)
        self.register("markdown", MarkdownReporter)  # Alias


# Global registry instance
_reporter_registry = ReporterRegistryImpl()


def get_reporter_registry() -> ReporterRegistryImpl:
    """Get global reporter registry instance."""
    return _reporter_registry


def register_reporter(format_name: str, reporter_class: type[Reporter]) -> None:
    """Register a reporter in the global registry."""
    _reporter_registry.register(format_name, reporter_class)
