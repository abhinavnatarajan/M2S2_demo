"""Logging support for multiprocessing.Pool."""

import logging
import multiprocessing.pool
from collections.abc import Callable, Iterable
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from typing import Any

__all__ = ["PoolWithLogger"]


def _init_worker(worker_log_queue: Queue, initializer: Callable, initargs: Iterable) -> None:
	"""Initialize a spawned worker using only pickleable arguments."""
	if worker_log_queue is not None:
		logging.captureWarnings(capture=True)
		worker_logger = logging.getLogger()
		worker_logger.handlers.clear()
		worker_logger.setLevel(logging.DEBUG)
		worker_logger.addHandler(QueueHandler(worker_log_queue))
	if callable(initializer):
		initializer(*initargs)


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
		# Forking after Polars has started its threads can deadlock workers.
		context = multiprocessing.get_context("spawn")
		self.manager = context.Manager()
		worker_log_queue = None
		if logger is not None:
			worker_log_queue = self.manager.Queue()
			self._log_listener = QueueListener(
				worker_log_queue,
				*logger.handlers,
				respect_handler_level=True,
			)
			self._log_listener.start()

		super().__init__(
			processes,
			_init_worker,
			(worker_log_queue, initializer, tuple(initargs)),
			maxtasksperchild,
			context=context,
		)

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
