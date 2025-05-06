from typing import Tuple, Dict
from numpy import  ndarray, int8, integer, floating, array, empty, concatenate, full, argsort, nonzero, repeat, zeros

from .bp_generator import BPGen, register_generator


@register_generator('demons')
class GeneratorDemons(BPGen):
    """
    Callable object

    Generates sequences based on length_dict,
    some sequences may be "overlapped": two sequences generated as one combined
    and then split by an overlap of 1...max_overlap elements,
    that create situation then demon can start with same symbols as other has ended,
    without braking rules of matrix pathing.
    Does not check length, so it may create sequences longer then maximum possible on given matrix.

    matrix:
        square 2d numpy array
    demons_lengths:
        1d array, where demons_length[idx] is amount of demons of length idx,
        may also be dict: {length: count, ...}
    overlap_chance:
        probability of overlapping, default=0.4
    max_overlap:
        maximum overlap length, default=2
    Returns:
        tuple with demons as arrays
    """
    _initialized:bool = False

    def _warm_up(self):
        try:
            self._initialized = True
            self._gen_single_sequence(1, array([[1]], dtype=int8), self.rng)
        except Exception as e:
            self._initialized = False
            raise RuntimeError(f"Error while initialization demon generator occurred:\n    {e}") from e

    def __call__(self,
                 matrix: ndarray, demons_lengths: array|Dict, overlap_chance: float = 0.4, max_overlap: int = 3
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
        if not isinstance(overlap_chance, (float, int, integer, floating)):
            raise TypeError("overlap_chance must be float (or convertable to float)")
        if not isinstance(max_overlap, (int, integer)):
            raise TypeError("max_overlap must be int (or convertable to int)")

        if not self._initialized: self._warm_up()
        if isinstance(demons_lengths, dict):
            lengths = concatenate([
                full(count, length, dtype=int8)
                for length, count in demons_lengths.items()])
        else:
            indices = nonzero(demons_lengths)[0]
            lengths = repeat(indices, demons_lengths[indices])

        # if any(i > matrix.size for i in lengths):
        #     raise ValueError(f"Length of some of demons is longer that maximum possible ({matrix.size}): {lengths}")

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
                overlap_leng = self.rng.integers(1, max_o+1) if max_o == 0 else 1

                total_len = leng1 + leng2 - overlap_leng  # noqa (numpy int vs python int)
                combined = self._gen_single_sequence(total_len, matrix, self.rng)

                if isinstance(combined, int) and combined == -1:
                    # generate sequences separately if attempt of generating combined failed
                    seq1 = self._gen_single_sequence(leng1, matrix, self.rng)
                    seq2 = self._gen_single_sequence(leng2, matrix, self.rng)
                    sequences[idx - 2] = seq1
                    sequences[idx - 1] = seq2
                else:
                    # split combined sequences if combined generated successfully
                    seq1 = combined[:leng1]
                    seq2 = combined[leng1 - overlap_leng:]  # noqa (numpy int vs python int)
                    sequences[idx - 2] = seq1
                    sequences[idx - 1] = seq2
            else:
                # generate single sequence normally
                seq = self._gen_single_sequence(leng1, matrix, self.rng)
                sequences[idx - 1] = seq

        # print(sequences)
        sorted_indices = argsort([seq.size for seq in sequences])
        sequences = sequences[sorted_indices]

        return tuple(sequences)

    @staticmethod
    # @njit(cache=True)
    def _gen_single_sequence(length: int, matrix: ndarray, rng) -> ndarray | int:
        """
        Generate single sequence for given matrix,
        if impossible return empty -1 to call function (for cases when overlapping)
        """

        size = matrix.shape[0]
        max_candidates = size - 1

        demon = empty(length, dtype=int8)
        used = zeros((size, size), dtype='bool')    # boolean mask of used cells
        candidates = empty((max_candidates, 2), dtype=int8)

        # attempt limitation just to be sure,
        # none of task_factory modes provides such specs for demons, but in case of manual creation return None on fail
        max_attempts = 100
        for att in range(max_attempts):
            used.fill(False)    # reset mask state

            i = rng.integers(0, size)
            j = rng.integers(0, size)

            demon[0] = matrix[i, j]
            used[i, j] = True
            curr_dir = bool(rng.integers(0, 2))

            valid = True

            for t in range(1, length):
                count = 0

                if curr_dir:
                    for cj in range(size):
                        if not used[i, cj]:
                            candidates[count, 0] = i
                            candidates[count, 1] = cj
                            count += 1
                else:
                    for ci in range(size):
                        if not used[ci, j]:
                            candidates[count, 0] = ci
                            candidates[count, 1] = j
                            count += 1

                if count == 0:
                    valid = False
                    break

                idx = rng.integers(0, count)
                i, j = candidates[idx, 0], candidates[idx, 1]
                demon[t] = matrix[i, j]
                used[i, j] = True
                curr_dir = not curr_dir

            if valid:
                return demon

        # return empty(length, dtype=int8)
        return -1
