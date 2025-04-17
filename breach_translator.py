BREACH_SYMBOLS_TO_INT = {
    # from basic game
    "1C": 1,
    "55": 2,
    "BD": 3,
    "E9": 4,
    "7A": 5,
    "FF": 6,
    # from dlc
    "X9": 7,
    "XX": 8,
    "XH": 9,
    "IX": 10,
    "XR": 11,
}
BREACH_INT_TO_SYMBOL = {v: k for k, v in BREACH_SYMBOLS_TO_INT.items()}


def map_symbol_to_int(symbol: str) -> int:
    """
    Returns integer value corresponding to breach protocol hex symbol
    :param symbol: hex as string
    :return: corresponding int
    """
    try:
        return BREACH_SYMBOLS_TO_INT[symbol]
    except KeyError:
        raise ValueError(f"Unknown symbol '{symbol}'. Must be one of: {list(BREACH_SYMBOLS_TO_INT.keys())}")


def map_int_to_symbol(value: int) -> str:
    """
    Returns hex symbol corresponding to given int in breach protocol
    :param value: integer value
    :return: corresponding hex symbol as string
    """
    try:
        return BREACH_INT_TO_SYMBOL[value]
    except KeyError:
        raise ValueError(f"Unknown integer '{value}'. Must be one of: {list(BREACH_INT_TO_SYMBOL.keys())}")