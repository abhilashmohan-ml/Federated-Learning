"""
Structured logging configuration using structlog + Python's stdlib logging.

WHY STRUCTURED LOGGING?
-----------------------
Traditional logging produces plain text like:
    "2026-07-04 10:00:01 INFO  Site 1 round completed"

Structured logging produces JSON like:
    {"level":"info","event":"round_complete","site":"site_1","round_id":3,"timestamp":"2026-07-04T10:00:01Z"}

The JSON format is much easier for:
  - Log aggregation systems (Datadog, ELK, Splunk) to parse and index
  - Filtering and searching: find all logs where round_id=3
  - Audit trails: every field is explicit, not buried in a sentence

We use `structlog`, which wraps Python's built-in `logging` module and
adds the structured (key=value) style.

PYTHON CONCEPT: Module vs function
  - This file defines TWO things: a function (`configure_logging`) called
    ONCE at application startup to configure logging globally, and a helper
    function (`get_logger`) called in each module that needs to log.
"""

import logging   # Python's built-in logging module
import os        # os.getenv reads environment variables
import structlog # third-party structured logging library


def configure_logging() -> None:
    """
    Configure the global logging system. Call this ONCE at app startup.

    HOW IT WORKS
    ------------
    1. Read the LOG_LEVEL environment variable (INFO by default).
       Levels in order of severity: DEBUG < INFO < WARNING < ERROR < CRITICAL
       Setting INFO means DEBUG messages are hidden; only INFO and above are shown.

    2. Configure Python's built-in `logging` with a simple format (just the
       message — structlog adds all the rich context).

    3. Configure structlog with a "processor pipeline" — a chain of functions
       that each transform the log record before it is finally emitted.

    PYTHON CONCEPT: None return type
      `-> None` means this function does not return any value.
      It only has side effects (changing global state).
    """
    # Read the log level from the environment, default to "INFO" if not set.
    # .upper() converts "info" or "Info" to "INFO" for consistency.
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure Python's stdlib logging with a plain format.
    # structlog will build the actual message string, so we just use %(message)s.
    logging.basicConfig(
        format="%(message)s",                              # only print the message itself
        level=getattr(logging, log_level, logging.INFO),  # convert string to log level int
    )

    # Configure structlog with a pipeline of processors.
    # Each processor takes the log record, modifies it, and passes it to the next.
    structlog.configure(
        processors=[
            # 1. Merge context variables set with structlog.contextvars.bind_contextvars()
            #    (useful for adding request IDs to every log in an async web handler)
            structlog.contextvars.merge_contextvars,

            # 2. Add the log level name (e.g. "info", "warning") to the record
            structlog.processors.add_log_level,

            # 3. Add an ISO-8601 timestamp (e.g. "2026-07-04T10:00:01.234Z")
            structlog.processors.TimeStamper(fmt="iso"),

            # 4. If a stack trace was captured, format it as a string
            structlog.processors.StackInfoRenderer(),

            # 5. If an exception was captured, format it as a string
            structlog.processors.format_exc_info,

            # 6. Final step: render everything as a JSON string ready for output
            structlog.processors.JSONRenderer(),
        ],
        # Only log records at or above log_level will be emitted
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,                           # use a plain Python dict for context
        logger_factory=structlog.PrintLoggerFactory(), # output to stdout (captured by Docker)
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for a specific module.

    Usage in any module:
        from shared.utils.logging_config import get_logger
        log = get_logger(__name__)      # __name__ is the module's full name
        log.info("round_started", round_id=1, site="site_1")

    This produces JSON output like:
        {"level":"info","event":"round_started","round_id":1,"site":"site_1","timestamp":"..."}

    PYTHON CONCEPT: __name__
      In Python, every module has a built-in variable called __name__ that
      holds the module's fully qualified name, e.g. "client.comms.fl_client".
      Passing it to get_logger() tags every log line with which module produced it.

    Parameters
    ----------
    name : str
        The module name, typically passed as __name__.

    Returns
    -------
    structlog.BoundLogger
        A logger instance that emits structured JSON log lines.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
