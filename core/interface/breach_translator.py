from numpy import ndarray, vectorize, integer
from core.base_setup import mapper_to_int, mapper_to_str

__all__ = ['map_breach']


# Note: Integer values are not actual translations of the hex codes — only codes for internal use
def map_breach(arraylike):
    """
    Automatically maps all elements in a 2d list or 2d ndarray between breach protocol symbols and their integer values

    :param arraylike: 2d list or array of symbols (str) or values (int)
    :return: 2d list of array of same shape as input with converted values (str <-> int)
    :raises ValueError: if input contains unknown values
    :raises TypeError: if input is empty or not consistent in types
        """

    # if isinstance(arraylike, ndarray):
    #     if arraylike.ndim != 2 or arraylike.size == 0:
    #         raise ValueError("Input array must be 2D and non-empty")
    #     first = arraylike[0, 0]
    # else:
    #     if not arraylike or not arraylike[0]:
    #         raise ValueError("Input list is empty or improperly structured")
    #     first = arraylike[0][0]

    if not arraylike[0][0]:
        raise ValueError("Input is empty")

    if isinstance(arraylike, ndarray):
        if arraylike.ndim != 2 or arraylike.size == 0:
            raise ValueError("Input array must be 2D and non-empty")
        first = arraylike.flat[0]
    else:
        first_row = arraylike[0]
        if isinstance(first_row, ndarray):
            if first_row.size == 0:
                raise ValueError("First row is empty")
            first = first_row[0]
        else:
            if not first_row:
                raise ValueError("First row is empty")
            first = first_row[0]

    if isinstance(first, str):
        mapper = mapper_to_int
    elif isinstance(first, (int, integer)):
        mapper = mapper_to_str
    else:
        raise TypeError(f"Unsupported type: {type(first).__name__}. Must be int or str")

    try:
        if isinstance(arraylike, ndarray):
            converted = vectorize(mapper)(arraylike)
        else:
            # converted = [[mapper(cell) for cell in row] for row in arraylike]
            converted = [
                [mapper(cell) for cell in row]
                if isinstance(row, (list, tuple)) else [mapper(cell)
                                                        for cell in row]
                for row in arraylike
            ]
    except KeyError as e:
        raise ValueError(f"Unknown value: {e.args[0]!r}")

    return converted


