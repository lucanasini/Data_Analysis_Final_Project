"""
constants.py
============
Constants and default variable lists.
"""


CLINICAL_COLS = ["BM.PB", "Gender", "Source", "tissue.mf"]

# Supported activations
SUPPORTED_ACTIVATIONS = ("relu", "leakyrelu", "sigmoid", "tanh", "softplus")

# Supported optimizers
SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd")

# Supported devices
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")
