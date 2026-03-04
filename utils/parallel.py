"""Logging support for multiprocessing.Pool."""

import logging
import multiprocessing.pool
from collections.abc import Callable, Iterable
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.managers import SyncManager
from typing import Any

__all__ = ["PoolWithLogger"]


class PoolWithLogger(multiprocessing.pool.Pool):
    """Subclass of multiprocessing.pool.Pool with worker logging."""

    def __init__(
        self,
        processes: int | None = None,
        initializer: Callable | None = None,
        initargs: Iterable[Any] = (),
        maxtasksperchild: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the PoolWithLogger.

        The arguments are passed on to multiprocessing.pool.Pool, with the exception of logger.
        The logger is connected to a QueueListener, which listens to a shared logging queue.
        When each worker is initialized with the initializer callable, the root logger
        of the worker process is configured to send all its  messages to the shared logging queue.
        """
        self.manager = multiprocessing.Manager()
        if logger is not None:
            worker_log_queue = self.manager.Queue()
            self._log_listener = QueueListener(
                worker_log_queue,
                *logger.handlers,
                respect_handler_level=True,
            )
            self._log_listener.start()

            def init_worker_logger() -> None:
                # this function is called in each worker process
                # it creates a logger that logs everything
                # and sends it to the shared queue
                logging.captureWarnings(capture=True)
                worker_logger = logging.getLogger()
                worker_logger.handlers.clear()
                worker_logger.setLevel(logging.DEBUG)
                worker_logger.addHandler(QueueHandler(worker_log_queue))

        def new_init(*args) -> None:
            # this function is the initialiser for the worker processes
            if logger is not None:
                # start a logger and connect it to the shared logging queue
                # if a global logger is provided
                init_worker_logger()
            if callable(initializer):
                # if an initialiser function is provided, call it
                initializer(*args)

        super().__init__(processes, new_init, initargs, maxtasksperchild)

    def get_manager(self) -> SyncManager:
        """Return the SyncManager instance that handles the shared logging queue."""
        return self.manager

    def close(self) -> None:
        """Close the pool and stop the logging queue listener."""
        # need to close the log listener to prevent memory leaks
        if getattr(self, "_log_listener", None) is not None:
            self._log_listener.stop()
        super().close()

    def terminate(self) -> None:
        """Terminate the pool and stop the logging queue listener."""
        # need to close the log listener to prevent memory leaks
        if getattr(self, "_log_listener", None) is not None:
            self._log_listener.stop()
        super().terminate()
