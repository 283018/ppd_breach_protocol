from numpy import ndarray, array, concatenate, full
from core.base_setup import BASE_INTS, DLC_INTS

from .bp_generator import BPGen, register_generator


@register_generator('matrix')
class GeneratorMatrix(BPGen):
    def __call__(self, size:int, mode:int=0) -> ndarray:
        """
        Generates breach protocol matrix of integer coded symbols.

        :param size: size of square matrix
        :param mode: generation mode (determines symbol sets and frequency)
                        0: simulates standard minigame, uses only base game symbols with equal chances,
                            recommended sizes: [3-6]
                        1: simulates dlc minigame, uses mostly dlc symbols, with small chances for base game symbols
                            recommended sizes: [5-8]
                        2: uses full set (base + dlc), slowly decrease chances of appearing for each new symbol, add step for dlc
                            recommended sizes: [6-12]
                        3: uses full set with equal chances for all
                            recommended sizes: (>10)
                        4: "good luck xD":
                            not recommended
        :return: square matrix as nested 2d list
        """
        # if not isinstance(size, int) or size <= 0:
        #     raise ValueError(f"Invalid length: {size}. Must be a positive integer.")

        base_range = BASE_INTS
        dlc_range = DLC_INTS


        if mode == 0:
            # just base game symbols, simulates base game bp
            final_range = base_range
            range_leng = final_range.size
            final_dists = full(range_leng, 1/range_leng)

        elif mode ==  1:
            # mostly dlc symbols, simulates dlc bp
            # weights of 1 for dlc, 0.4 for base
            base_to_add_leng = min(size // 3, base_range.size)
            dlc_leng = dlc_range.size
            weight_base = 0.5
            weight_dlc = 1.0

            final_range = concatenate((
                dlc_range,
                self.rng.choice(base_range, base_to_add_leng, replace=False)))


            total_weight = dlc_leng * weight_dlc + base_to_add_leng * weight_base
            prob_base = weight_base /  total_weight
            prob_dlc = weight_dlc / total_weight

            final_dists = concatenate((
                full(dlc_leng, prob_dlc),
                full(base_to_add_leng, prob_base)))

        elif mode ==  2:
            # slowly decreasing chances for newer characters with step for dlc
            final_range = concatenate((base_range, dlc_range))

            final_dists = concatenate((
                1.0 / base_range + 2,
                1.0 / dlc_range + 1))
            final_dists /= final_dists.sum()

        elif mode ==  3:
            # equal chances for all symbols base+dlc (recommended for large matrices only)
            final_range = concatenate((base_range, dlc_range))
            final_dists = None

        elif mode == 4:
            final_range = array(i for i in range(1, 99))
            final_dists = None

        else:
            raise ValueError(f"Invalid mode: {mode}")


        # matrix = self.rng.choice(final_range, size=(size, size), p=final_dists)
        return self.rng.choice(final_range, size=(size, size), p=final_dists)








# if __name__ == "__main__":
#     rng1 = default_rng()
#     gen1 = GeneratorMatrix(rng1)
#
#     mat1 = gen1(4)
#     print(mat1)
