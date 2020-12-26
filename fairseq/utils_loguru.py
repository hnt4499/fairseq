"""Loguru utils"""

# List of files in `fairseq_cli` that use logging. Any files other than these
# attempting to use logging will have their logger with the same file name
# (i.e., `__name__`).
name_list = ["eval_lm", "generate", "hydra_train", "interactive", "preprocess",
             "train", "validate"]


def loguru_name_patcher(record):
    filename = record["file"].name  # filename, e.g., `train.py`
    name = ".".join(filename.split(".")[:-1])  # remove the ".py" part
    if name in name_list:
        name = f"fairseq_cli.{name}"  # legacy name, e.g., `fairseq_cli.train`
    record["extra"].update(name=name)


def loguru_reset_logger(logger):
    """Remove all handlers"""
    handlers_count = logger._core.handlers
    for _ in range(len(handlers_count)):
        logger.remove()


class LoguruLevels:
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def loguru_set_level(logger, level):
    """Set level of all handlers of the provided logger. Note that this
    implementation is very non-standard. Avoid using in any case."""
    for handler in logger._core.handlers.values():
        handler._levelno = level


def get_effective_level(logger):
    """Get effective level of the logger by finding the smallest level among
    all handlers."""
    levels = []
    for handler in logger._core.handlers.values():
        levels.append(handler.levelno)
    return min(levels)


def loguru_emit_some_handlers(logger, handler_ids, message, name,
                              level_id="INFO"):
    """Emit message using specific handlers while ignoring others. Currently
    only supports for non-colorized messages.

    Parameters
    ----------
    logger : loguru._logger.Logger
        Loguru logger.
    handler_ids : list of int
        List of handler IDs to deal with.
    message : str
        Message to emit.
    name : str
        Logger name.
    level_id : str
        Level name.

    """
    from loguru._recattrs import RecordLevel
    from loguru._datetime import aware_now

    core = logger._core
    level_name, level_no, _, level_icon = core.levels[level_id]

    for handler_id in handler_ids:
        handler = core.handlers[handler_id]
        log_record = {
            "message": message,
            "level": RecordLevel(level_name, level_no, level_icon),
            "exception": None,
            "time": aware_now(),
            "extra": {"name": name},
        }
        handler.emit(log_record, level_id, from_decorator=False, is_raw=False,
                     colored_message=None)
