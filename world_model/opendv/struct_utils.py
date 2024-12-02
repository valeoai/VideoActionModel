import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]


def save_struct(
    array: np.ndarray,
    filename: PathLike,
    max_value: Optional[int] = None,
) -> None:
    """Save sequence using struct with shape information."""
    assert array.ndim <= 3, "Array must have 3 dimensions"
    dim = array.ndim
    length = rows = cols = 1
    if array.ndim == 1:
        length = len(array)
    elif array.ndim == 2:
        rows, cols = array.shape
    elif array.ndim == 3:
        length, rows, cols = array.shape
    k = max_value or array.max()

    # Choose minimal dtype
    if k <= 255:
        dtype = np.uint8
    elif k <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32

    array = array.astype(dtype)

    with open(filename, "wb") as f:
        f.write(struct.pack("QQQQQ", dim, length, rows, cols, k))
        array.tofile(f)


def load_struct(filename: PathLike) -> np.ndarray:
    """Load sequence using struct with shape information."""
    with open(filename, "rb") as f:
        dim, length, rows, cols, k = struct.unpack("QQQQQ", f.read(40))

        if k <= 255:
            dtype = np.uint8
        elif k <= 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32

        arr = np.fromfile(f, dtype=dtype).reshape(length, rows, cols)
        if dim == 1:
            return arr[:, 0, 0]
        elif dim == 2:
            return arr[:, :, 0]
        return arr
