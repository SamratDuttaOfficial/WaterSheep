"""WaterSheep: a fast decision model with typed, calibrated answers."""
__version__ = "0.1.0"


def __getattr__(name):
    # `from watersheep import WaterSheep` without importing torch for every submodule
    if name == "WaterSheep":
        from .infer import WaterSheep
        return WaterSheep
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
