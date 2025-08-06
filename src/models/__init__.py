"""Neural network models for curve fitting research."""

from .architectures import BaseModel, LinearModel, ShallowNetwork, DeepNetwork, ModelFactory, ModelCheckpoint

__all__ = ['BaseModel', 'LinearModel', 'ShallowNetwork', 'DeepNetwork', 'ModelFactory', 'ModelCheckpoint']