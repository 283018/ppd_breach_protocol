#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <omp.h>    // NOLINT


struct DFSResult {
    std::unique_ptr<int[]> path;
    int score{};
    int length{};
};

DFSResult processColumn(
    const int start_col,
    const int* matrix,
    const int* demons_array,
    const int* demons_lengths,
    const int* demons_costs,
    const int buffer_size,
    const int n,
    const int max_score,
    const int num_demons,
    const int init_stack_size,
    const int padded_demon_length,
    const bool enable_pruning,
    const double time_limit
) {

    bool time_expired = false;
    const std::chrono::steady_clock::time_point start_time = time_limit > 0.0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // allocating stack
    const int max_stack = init_stack_size;
    std::unique_ptr<int[]> stack_path(new int[max_stack * buffer_size * 2]);
    std::unique_ptr<int[]> stack_buff(new int[max_stack * buffer_size]);
    std::unique_ptr<bool[]> stack_activ(new bool[max_stack * num_demons]);
    std::unique_ptr<int[]> stack_score(new int[max_stack]);
    std::unique_ptr<bool[]> stack_used(new bool[max_stack * n * n]);
    std::unique_ptr<int[]> stack_length(new int[max_stack]);

    // best path
    std::unique_ptr<int[]> best_path(new int[buffer_size * 2]);
    int best_score = 0;
    int best_path_length = 0;
    std::fill_n(best_path.get(), buffer_size * 2, -1);

    // init pos setup
    constexpr int start_r = 0;
    const int start_c = start_col;
    const int start_symbol = matrix[start_r * n + start_c];

    // first push onto stack
    std::fill_n(stack_path.get(), max_stack * buffer_size * 2, -1);
    stack_path[0 * buffer_size * 2 + 0 * 2] = start_r;
    stack_path[0 * buffer_size * 2 + 0 * 2 + 1] = start_c;

    std::fill_n(stack_buff.get(), max_stack * buffer_size, -1);
    stack_buff[0 * buffer_size + 0] = start_symbol;

    for (int d = 0; d < num_demons; ++d) {
        const bool active = (demons_lengths[d] == 1 && demons_array[d * padded_demon_length + 0] == start_symbol);
        stack_activ[0 * num_demons + d] = active;
    }

    std::fill_n(stack_used.get(), max_stack * n * n, false);
    stack_used[0 * n * n + start_r * n + start_c] = true;

    int initial_score = 0;
    for (int d = 0; d < num_demons; ++d) {
        if (stack_activ[0 * num_demons + d]) {
            initial_score += demons_costs[d];
        }
    }
    stack_score[0] = initial_score;
    stack_length[0] = 1;

    int pointer = 1;

    // main DFS loop
    while (pointer > 0) {
        --pointer;
        const int curr_ptr = pointer;

        if (time_limit > 0.0 && !time_expired) {
            auto current_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            if (elapsed >= time_limit) {
                time_expired = true;
            }
        }

        const int curr_len = stack_length[curr_ptr];
        const int curr_score = stack_score[curr_ptr];

        // update best path
        if ((curr_score > best_score ||
                    (!enable_pruning &&
                    curr_score == best_score &&
                    curr_len < best_path_length)) &&
                    curr_len <= buffer_size
            ) [[unlikely]] {
                best_score = curr_score;
                best_path_length = curr_len;
                for (int i = 0; i < curr_len; ++i) {
                    best_path[i * 2] = stack_path[curr_ptr * buffer_size * 2 + i * 2];
                    best_path[i * 2 + 1] = stack_path[curr_ptr * buffer_size * 2 + i * 2 + 1];
                }
                if (enable_pruning && best_score == max_score) [[likely]] {
                    break;
                }
        }

        // if process children (with time limit)
        if (curr_len < buffer_size && !time_expired) {
            const int next_step = curr_len + 1;
            const int last_r = stack_path[curr_ptr * buffer_size * 2 + (curr_len - 1) * 2];
            const int last_c = stack_path[curr_ptr * buffer_size * 2 + (curr_len - 1) * 2 + 1];

            const bool row_fixed = (next_step % 2 == 0);
            const bool col_fixed = !row_fixed;

            for (int idx = 0; idx < n; ++idx) {
                const int r = col_fixed ? last_r : idx;
                const int c = col_fixed ? idx : last_c;

                if (!stack_used[curr_ptr * n * n + r * n + c]) {
                    const int new_ptr = pointer;
                    if (new_ptr >= max_stack) [[unlikely]] throw std::runtime_error("Stack overflow!");;

                    // copy path
                    std::copy_n(
                        &stack_path[curr_ptr * buffer_size * 2],
                        buffer_size * 2,
                        &stack_path[new_ptr * buffer_size * 2]
                    );
                    stack_path[new_ptr * buffer_size * 2 + curr_len * 2] = r;
                    stack_path[new_ptr * buffer_size * 2 + curr_len * 2 + 1] = c;
                    // copy buffer
                    std::copy_n(
                        &stack_buff[curr_ptr * buffer_size],
                        buffer_size,
                        &stack_buff[new_ptr * buffer_size]
                    );
                    stack_buff[new_ptr * buffer_size + curr_len] = matrix[r * n + c];
                    // copy used mask
                    std::copy_n(
                        &stack_used[curr_ptr * n * n],
                        n * n,
                        &stack_used[new_ptr * n * n]
                    );
                    stack_used[new_ptr * n * n + r * n + c] = true;
                    // copy active demons
                    std::copy_n(
                        &stack_activ[curr_ptr * num_demons],
                        num_demons,
                        &stack_activ[new_ptr * num_demons]
                    );

                    int new_score = curr_score;
                    const int new_len = next_step;

                    // check demon matches
                    for (int d = 0; d < num_demons; ++d) {
                        if (!stack_activ[new_ptr * num_demons + d]) {
                            const int k = demons_lengths[d];
                            if (new_len >= k) {
                                bool match = true;
                                for (int j = 0; j < k; ++j) {
                                   if (stack_buff[new_ptr * buffer_size + (new_len - k + j)] != demons_array[d*padded_demon_length + j]) {
                                       match = false;
                                       break;
                                   }
                                }
                                if (match) {
                                    stack_activ[new_ptr * num_demons + d] = true;
                                    new_score += demons_costs[d];
                                }
                            }
                        }
                    }

                    // optional pruning
                    bool do_push = true;
                    if (enable_pruning) [[likely]] {
                        int rem = 0;
                        for (int d = 0; d < num_demons; ++d) {
                            if (!stack_activ[new_ptr * num_demons + d]) {
                                rem += demons_costs[d];
                            }
                        }
                        do_push = new_score + rem > best_score;
                    }

                    if (do_push) [[likely]] {
                        stack_score[new_ptr] = new_score;
                        stack_length[new_ptr] = new_len;
                        ++pointer;
                    }
                }
            }
        }
    }

    auto path = std::make_unique<int[]>(best_path_length * 2);
    std::copy_n(best_path.get(), best_path_length * 2, path.get());

    return {std::move(path), best_score, best_path_length};
}



namespace py = pybind11;

DFSResult runParallelColumns(
    const int* matrix,
    const int* demons_array,
    const int* demons_lengths,
    const int* demons_costs,
    const int buffer_size,
    const int n,
    const int max_score,
    const int num_demons,
    const int init_stack_size,
    const int padded_demon_length,
    const bool enable_pruning,
    const double time_limit
) {

    auto* all_results = new DFSResult[n];

    {
        py::gil_scoped_release release;     // in theory that is required, in practise - not really, in my opinion let it be
        #pragma omp parallel for
        for (int start_col = 0; start_col < n; ++start_col) {
            all_results[start_col] = processColumn(
                start_col,
                matrix,
                demons_array,
                demons_lengths,
                demons_costs,
                buffer_size,
                n,
                max_score,
                num_demons,
                init_stack_size,
                padded_demon_length,
                enable_pruning,
                time_limit
            );
        }
    }

    int best_index = 0;
    for (int i = 1; i < n; ++i) {
        if (all_results[i].score > all_results[best_index].score ||
            (all_results[i].score == all_results[best_index].score &&
             all_results[i].length < all_results[best_index].length)) {
            best_index = i;
             }
    }

    auto best_result = std::move(all_results[best_index]);
    delete[] all_results;

    return best_result;
}




auto processBreach_fromNumpy(
    py::array_t<int32_t> matrix_np,
    py::array_t<int32_t> demons_array_np,
    py::array_t<int32_t> demons_lengths_np,
    py::array_t<int32_t> demons_costs_np,
    const int buffer_size,
    const int n,
    const int max_score,
    const int num_demons,
    const int init_stack_size,
    const bool enable_pruning,
    const double time_limit
) -> py::array_t<int32_t> {

    // simple array validation
    if (!matrix_np.request().ptr || !demons_array_np.request().ptr ||
        !demons_lengths_np.request().ptr || !demons_costs_np.request().ptr) {
        throw std::runtime_error("Input arrays must not be empty or uninitialized.");
        }

    matrix_np = py::array_t(matrix_np);
    demons_array_np = py::array_t(demons_array_np);
    demons_lengths_np = py::array_t(demons_lengths_np);
    demons_costs_np = py::array_t(demons_costs_np);


    const auto padded_demon_length = static_cast<int>(demons_array_np.shape()[1]);

    const int* matrix = matrix_np.data();
    const int* demons_array = demons_array_np.data();
    const int* demons_lengths = demons_lengths_np.data();
    const int* demons_costs = demons_costs_np.data();

    // call parallel function
    const DFSResult best_result = runParallelColumns(
        matrix,
        demons_array,
        demons_lengths,
        demons_costs,
        buffer_size,
        n,
        max_score,
        num_demons,
        init_stack_size,
        padded_demon_length,
        enable_pruning,
        time_limit
    );

    // translating back to numpy
    py::array_t<int32_t> path_array({best_result.length, 2});
    const auto path_data = static_cast<int32_t*>(path_array.request().ptr);
    std::copy_n(
        best_result.path.get(),
        2 * best_result.length,
        path_data
    );

    return path_array;
}


PYBIND11_MODULE(breach_solver_cpp, m) {
    m.doc() = "Solve breach protol matrix algorythm stays same as in python/numba version, require padding and pre-processing";

    m.def("brute_force", &processBreach_fromNumpy,
        py::arg("matrix_np"),
        py::arg("demons_array_np"),
        py::arg("demons_lengths_np"),
        py::arg("demons_costs_np"),
        py::arg("buffer_size"),
        py::arg("n"),
        py::arg("max_score"),
        py::arg("num_demons"),
        py::arg("init_stack_size"),
        py::arg("enable_pruning"),
        py::arg("time_limit")
        );
}

