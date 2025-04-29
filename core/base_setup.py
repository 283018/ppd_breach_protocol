from numpy import array


# File: core/base_setup.py
__all__ = [
    'HEX_MAP', 'HEX_MAP_REVERSE', 'BASE_INTS', 'DLC_INTS', 'BASE_HEXS', 'DLC_HEXS', 'mapper_to_str', 'mapper_to_int'
]


# core mapping definitions
# keys only up to 99 to keep 2-char format (update generators if needed)
HEX_MAP = {
    0: " ▧",                                                # 0 mapping (only for displaying)
    1: "1C", 2: "55", 3: "BD", 4: "E9", 5: "7A", 6: "FF",   # base game
    7: "X9", 8: "XX", 9: "XH", 10: "IX", 11: "XR",          # dlc
}
HEX_MAP_REVERSE = {v: k for k, v in HEX_MAP.items()}


# keys/values arrays, base/dlc separated
base_keys = [k for k in HEX_MAP if 1 <= k <= 6]     # base keys (1-6)
dlc_keys = [k for k in HEX_MAP if 7 <= k <= 11]     # dlc keys (7-11)

BASE_INTS = array(base_keys, dtype='int8')
DLC_INTS = array(dlc_keys, dtype='int8')

BASE_HEXS = array([HEX_MAP[k] for k in base_keys], dtype='<U2')
DLC_HEXS = array([HEX_MAP[k] for k in dlc_keys], dtype='<U2')


# mappers
mapper_to_str = HEX_MAP.__getitem__
mapper_to_int = HEX_MAP_REVERSE.__getitem__




