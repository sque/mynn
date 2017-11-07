from typing import Union
import numpy as np

FloatOrArray = Union[np.ndarray, float]


BIG_FLOAT = np.float(1.0e+16)
SMALL_FLOAT = np.float(1.0e-16)
