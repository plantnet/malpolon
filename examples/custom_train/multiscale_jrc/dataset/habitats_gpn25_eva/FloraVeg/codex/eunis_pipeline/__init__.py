"""Offline, multi-head EUNIS habitat classification pipeline.

The package separates experiment configuration, metadata handling, neural
networks, optimisation, metrics, and artifact reporting so each can be tested
or replaced without changing the training entry point.
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
