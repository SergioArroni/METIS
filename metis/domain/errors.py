"""Domain-specific exceptions for error handling."""


class METISError(Exception):
    """Base exception for METIS framework."""

    pass


class ConfigError(METISError):
    """Raised when configuration is invalid or malformed."""

    def __init__(self, message: str, config_path: str | None = None):
        self.config_path = config_path
        full_message = f"Configuration error: {message}"
        if config_path:
            full_message += f" (in {config_path})"
        super().__init__(full_message)


class SchemaError(METISError):
    """Raised when data schema validation fails."""

    def __init__(self, message: str, column: str | None = None):
        self.column = column
        full_message = f"Schema error: {message}"
        if column:
            full_message += f" (column: {column})"
        super().__init__(full_message)


class RegistryError(METISError):
    """Raised when registry operations fail."""

    def __init__(self, message: str, registry_type: str | None = None, item_id: str | None = None):
        self.registry_type = registry_type
        self.item_id = item_id

        full_message = f"Registry error: {message}"
        if registry_type and item_id:
            full_message += f" ({registry_type} registry, item: {item_id})"
        elif registry_type:
            full_message += f" ({registry_type} registry)"

        super().__init__(full_message)


class PreprocessingError(METISError):
    """Raised when data preprocessing fails."""

    def __init__(
        self,
        message: str,
        step: str | None = None,
        original_error: Exception | None = None,
    ):
        self.step = step
        self.original_error = original_error

        full_message = f"Preprocessing error: {message}"
        if step:
            full_message += f" (step: {step})"
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"

        super().__init__(full_message)


class TypeCastingError(METISError):
    """Raised when type casting/transformation fails."""

    def __init__(
        self,
        message: str,
        column: str | None = None,
        expected_type: str | None = None,
        original_error: Exception | None = None,
    ):
        self.column = column
        self.expected_type = expected_type
        self.original_error = original_error

        full_message = f"type casting error: {message}"
        if column:
            full_message += f" (column: {column})"
        if expected_type:
            full_message += f" (expected type: {expected_type})"
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"

        super().__init__(full_message)
