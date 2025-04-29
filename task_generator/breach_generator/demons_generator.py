from typing import Tuple, Dict
from numpy import array, ndarray, empty, concatenate, full, argsort, nonzero, repeat

from .bp_generator import BPGen, register_generator


@register_generator('demons')
class GeneratorDemons(BPGen):
    def __call__(self,
                 matrix:ndarray, demons_lengths:array|Dict, overlap_chance:float = 0.4, max_overlap:int = 2
                 ) -> Tuple[array, ...]:
        """
        Generates sequences based on length_dict,
        some sequences may be "overlapped": two sequences generated as one combined
        and then split by an overlap of 1...max_overlap elements,
        that create situation then demon can start with same symbols as other has ended,
        without braking rules of matrix pathing.
        Does not check length, so it may create sequences longer then maximum possible on given matrix.

        :param matrix: square 2d numpy array
        :param demons_lengths: 1d array, where demons_length[idx] is amount of demons of length idx,
            may also be dict: {length: count, ...} for backward compatibility ig
        :param overlap_chance: probability of overlapping, default=0.4
        :param max_overlap: maximum overlap length, default=2
        :return: tuple with demons as arrays
        """
        if isinstance(demons_lengths, dict):
            lengths = concatenate([
                full(count, length, dtype='int8')
                for length, count in demons_lengths.items()])
        else:
            indices = nonzero(demons_lengths)[0]
            lengths = repeat(indices, demons_lengths[indices])

        self.rng.shuffle(lengths)

        total = lengths.size
        idx = 0
        sequences = empty(total, dtype=object)

        while idx < total:
            leng1 = int(lengths[idx])
            idx += 1

            # decide on overlap
            if idx < total and self.rng.random() < overlap_chance:
                leng2 = int(lengths[idx])
                idx += 1

                # generate overlap leng
                max_o = min(max_overlap, leng1, leng2)
                overlap_leng = self.rng.integers(1, max_o + 1)

                total_len = leng1 + leng2 - overlap_leng
                combined = self._gen_single_sequence(total_len, matrix)

                # split and save 2 sequences
                seq1 = combined[:leng1]
                seq2 = combined[leng1 - overlap_leng:]
                sequences[idx - 2] = seq1
                sequences[idx - 1] = seq2
            else:
                seq = self._gen_single_sequence(leng1, matrix)
                sequences[idx - 1] = seq

        
        sorted_indices = argsort([seq.size for seq in sequences])
        sequences = sequences[sorted_indices]
        
        return tuple(sequences)
    
    
    def _gen_single_sequence(self, length:int, matrix:ndarray) -> array:
        size = matrix.shape[0]
        demon = empty(length, dtype='int8')

        i, j = self.rng.integers(0, size, 2)
        demon[0] = matrix[i, j]

        # True corresponds to row, False to column
        curr_dir = self.rng.integers(0, 2, dtype=bool)

        for t in range(1, length):
            if curr_dir:
                next_j = self.rng.integers(0, size - 1, dtype='int8')
                next_j = next_j if next_j < j else next_j + 1
                demon[t] = matrix[i, next_j]
                j = next_j
            else:
                next_i = self.rng.integers(0, size - 1, dtype='int8')
                next_i = next_i if next_i < i else next_i + 1
                demon[t] = matrix[next_i, j]
                i = next_i

            curr_dir = not curr_dir

        return demon






# if __name__ == '__main__':
#     rng1 = default_rng()
#     gen1 = GeneratorDemons(rng1)
#
#     demons1 = gen1(array([i+1 for i in range(9)]).reshape((3, 3)), {1:2, 2:3})
#     print(demons1)