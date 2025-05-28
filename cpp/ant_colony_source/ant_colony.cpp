#include <optional>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <memory>

// only for conversion in python interface
#include <cstdint>      // NOLINT: not needed for c++20
#include <vector>
// #include <iostream>     // NOLINT: was here for testing

#include <omp.h>        // NOLINT: not needed for c++20

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;



struct SolCandidate {
    std::unique_ptr<std::pair<int, int>[]> path;
    int cost;
    size_t length;
    std::unique_ptr<int[]> buffer_seq;

    SolCandidate()
        : path(nullptr), cost(0), length(0), buffer_seq(nullptr) {}

    SolCandidate(std::unique_ptr<std::pair<int, int>[]> p, const int c, const size_t l, std::unique_ptr<int[]> b)
        : path(std::move(p)), cost(c), length(l), buffer_seq(std::move(b)) {}

    SolCandidate(SolCandidate&&) noexcept = default;
    SolCandidate& operator=(SolCandidate&&) noexcept = default;

    SolCandidate(const SolCandidate&) = delete;
    SolCandidate& operator=(const SolCandidate&) = delete;

    ~SolCandidate() = default;

    [[nodiscard]] SolCandidate clone() const {
        std::unique_ptr<std::pair<int, int>[]> new_path;
        if (path && length > 0) {
            new_path = std::make_unique<std::pair<int, int>[]>(length);
            std::copy_n(path.get(), static_cast<int>(length), new_path.get());
        }

        std::unique_ptr<int[]> new_buffer;
        if (buffer_seq && length > 0) {
            new_buffer = std::make_unique<int[]>(length);
            std::copy_n(buffer_seq.get(), static_cast<int>(length), new_buffer.get());
        }

        return {std::move(new_path), cost, length, std::move(new_buffer)};
    }
};


class Solver {
    const int* matrix;
    const int* demons;
    const int* demon_lengths;
    const int* demons_costs;
    const int num_demons;
    const int buffer_size;
    const int n;
    const int matrix_size;
    const int mat_sqr;
    const int max_demon_length;
    const double* heuristic;
    const double alpha;
    const double beta;
    const double evap_factor;
    const double q;
    alignas(32) std::unique_ptr<double[]> pheromone;

    std::mt19937 rng;
    std::unique_ptr<std::mt19937[]> ant_rngs;

    static int to_flat(const int r, const int c, const int n) { return r * n + c; }

    static bool compare_for_max(const SolCandidate& a, const SolCandidate& b) {
        if (a.cost < b.cost) return true;
        if (a.cost > b.cost) return false;
        return a.length > b.length;
    }

    static bool compare_for_sort(const SolCandidate& a, const SolCandidate& b) {
        if (a.cost > b.cost) return true;
        if (a.cost < b.cost) return false;
        return a.length < b.length;
    }

    void set_pheromone(const double def_value = 1.0) const {
        std::fill_n(pheromone.get(), mat_sqr, def_value);
    }

    std::optional<std::pair<int, int>> next_move(
        const std::pair<int, int>& last,
        const bool is_even,
        const bool* visited,
        std::mt19937& rng) const {
        const int max_candidates = n - 1;
        auto candidates = std::make_unique<std::pair<int, int>[]>(max_candidates);
        int candidate_count = 0;
        int r = last.first, c = last.second;

        if (is_even) {
            for (int col = 0; col < n; ++col) {
                if (col != c && !visited[to_flat(r, col, n)]) {
                    candidates[candidate_count++] = {r, col};
                }
            }
        } else {
            for (int row = 0; row < n; ++row) {
                if (row != r && !visited[to_flat(row, c, n)]) {
                    candidates[candidate_count++] = {row, c};
                }
            }
        }

        if (candidate_count == 0) [[unlikely]] return std::nullopt;

        auto scores = std::make_unique<double[]>(candidate_count);
        double total = 0.0;
        const int last_idx = to_flat(r, c, n);

        for (int i = 0; i < candidate_count; ++i) {
            const auto& [row, col] = candidates[i];
            const int idx = to_flat(row, col, n);
            const double tau = std::pow(pheromone[last_idx * matrix_size + idx], alpha);
            const double eta = std::pow(heuristic[idx], beta);
            scores[i] = tau * eta;
            total += scores[i];
        }

        std::optional<std::pair<int, int>> result;
        if (total <= 0.0) {
            std::uniform_int_distribution dist(0, candidate_count - 1);
            result = candidates[dist(rng)];
        } else {
            std::discrete_distribution dist(scores.get(), scores.get() + candidate_count);
            result = candidates[dist(rng)];
        }

        return result;
    }

    SolCandidate construct_solution(const int ant_idx) {
        auto path = std::make_unique<std::pair<int, int>[]>(buffer_size);
        auto buffer_vals = std::make_unique<int[]>(buffer_size);
        auto visited = std::make_unique<bool[]>(matrix_size);
        auto& local_rng = ant_rngs[ant_idx];

        std::uniform_int_distribution col_dist(0, n - 1);
        std::pair current(0, col_dist(local_rng));
        path[0] = current;
        visited[to_flat(current.first, current.second, n)] = true;
        buffer_vals[0] = matrix[current.first * n + current.second];
        size_t current_length = 1;
        bool is_even_step = false;

        while (current_length < buffer_size) {
            auto next = next_move(current, is_even_step, visited.get(), local_rng);
            if (!next) break;
            current = *next;
            path[current_length] = current;
            visited[to_flat(current.first, current.second, n)] = true;
            buffer_vals[current_length] = matrix[current.first * n + current.second];
            current_length++;
            is_even_step = !is_even_step;
        }

        int total_cost = 0;
        for (int d = 0; d < num_demons; ++d) {
            const int demon_len = demon_lengths[d];
            if (demon_len == 0) continue;
            const int* demon = &demons[d * max_demon_length];
            bool found = false;
            for (size_t i = 0; i + demon_len <= current_length; ++i) {
                bool match = true;
                for (int j = 0; j < demon_len; ++j) {
                    if (buffer_vals[i + j] != demon[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    found = true;
                    break;
                }
            }
            if (found) total_cost += demons_costs[d];
        }

        return {std::move(path), total_cost, current_length, std::move(buffer_vals)};
    }

    std::unique_ptr<SolCandidate[]> get_solutions_candidates(const int n_ants) {
        auto solutions = std::make_unique<SolCandidate[]>(n_ants);
        {
            py::gil_scoped_release release;
            #pragma omp parallel for
            for (int i = 0; i < n_ants; ++i) {
                solutions[i] = construct_solution(i);
            }
        }
        return solutions;
    }

    static SolCandidate update_best(const SolCandidate* current_best, const SolCandidate* candidates, const int num_candidates) {
        if (num_candidates == 0) [[unlikely]] {
            if (!current_best) throw std::runtime_error("No candidates provided");  // NOLINT: Always false for const inputs
            return current_best->clone();
        }

        const SolCandidate* best_it = &candidates[0];
        for (int i = 1; i < num_candidates; ++i) {
            if (compare_for_max(*best_it, candidates[i])) {
                best_it = &candidates[i];
            }
        }

        if (!current_best || compare_for_max(*current_best, *best_it)) {    // NOLINT: Always false for const inputs
            return best_it->clone();
        }

        return current_best->clone();
    }

    static SolCandidate* select_top_solutions(SolCandidate* solutions, const int num_solutions, const int n_ants, int* out_size) {
        const int k = std::max(1, n_ants / 2);
        *out_size = std::min(k, num_solutions);
        std::sort(solutions, solutions + num_solutions, [](const SolCandidate& a, const SolCandidate& b) {
            return compare_for_sort(a, b);
        });
        return solutions;
    }

    void update_pheromones(const SolCandidate* solutions, const int num_solutions) const {
        {
            py::gil_scoped_release release;
            #pragma omp parallel for
            for (int i = 0; i < mat_sqr; ++i) {
                pheromone[i] *= evap_factor;
            }
        }

        for (int s = 0; s < num_solutions; ++s) {
            const SolCandidate& sol = solutions[s];
            if (sol.length <= 1) continue;
            const double deposit = q * sol.cost / static_cast<double>(sol.length);
            for (size_t i = 0; i < sol.length - 1; ++i) {
                const int from = to_flat(sol.path[i].first, sol.path[i].second, n);
                const int to = to_flat(sol.path[i + 1].first, sol.path[i + 1].second, n);
                pheromone[from * matrix_size + to] += deposit;
            }
        }
    }


public:
    Solver(     // NOLINT: seed provided from python
        const int* matrix_, const int* demons_,
        const int* demon_lengths_, const int* demons_costs_,
        const int num_demons_, const int buffer_size_,
        const int n_, const int max_demon_length_,
        const double* heuristic_,
        const double alpha_, const double beta_,
        const double evaporation_, const double q_,
        const unsigned int seed)
      : matrix(matrix_), demons(demons_),
        demon_lengths(demon_lengths_), demons_costs(demons_costs_),
        num_demons(num_demons_), buffer_size(buffer_size_),
        n(n_), matrix_size(n * n), mat_sqr(matrix_size * matrix_size),
        max_demon_length(max_demon_length_),
        heuristic(heuristic_),
        alpha(alpha_), beta(beta_),
        evap_factor(1.0 - evaporation_), q(q_) {
            rng.seed(seed);
            pheromone = std::make_unique<double[]>(mat_sqr);
            set_pheromone();
    }

    SolCandidate run_ants(const int n_ants, const int n_iterations, const int stagnant_limit) {
        // setup of thread rngs
        ant_rngs = std::make_unique<std::mt19937[]>(n_ants);
        std::uniform_int_distribution<unsigned int> seed_dist(0, UINT_MAX);
        const unsigned int base_seed = seed_dist(rng);
        for (int i = 0; i < n_ants; ++i) {
            ant_rngs[i].seed(base_seed + i);
        }

        int no_improvements = 0;
        int best_cost = -1;

        const int no_improve_limit = stagnant_limit == 0 ? n * buffer_size * num_demons : stagnant_limit;
        const bool check_no_improvement = no_improve_limit > 0;

        const int effective_limit = n_iterations <= 0 ? INT_MAX : n_iterations;

        // std::cout
        // << "stagnant_limit = " << stagnant_limit << std::endl
        // << "no_improve_limit = " << no_improve_limit << std::endl
        // << "check_no_improvement = " << check_no_improvement << std::endl
        // << "n_iterations = " << n_iterations << std::endl
        // << "effective_limit = " << effective_limit << std::endl
        // << std::endl << std::flush;

        std::unique_ptr<SolCandidate> best_ptr;
        int i;
        for (i = 0; i < effective_limit; ++i) {

            // ants lets go
            auto solutions = get_solutions_candidates(n_ants);
            SolCandidate current_best = update_best(best_ptr.get(), solutions.get(), n_ants);

            // updating best, or counting no_improve
            if (current_best.cost > best_cost) {
                // std::cout << "iter " << i << " cost " << best_cost << std::endl << std::flush;
                best_cost = current_best.cost;
                no_improvements = 0;
            } else {
                ++no_improvements;
            }

            best_ptr = std::make_unique<SolCandidate>(std::move(current_best));

            if (check_no_improvement && no_improvements >= no_improve_limit) { break; }

            int top_size;
            const SolCandidate* top_solutions = select_top_solutions(solutions.get(), n_ants, n_ants, &top_size);
            update_pheromones(top_solutions, top_size);
        }


        // std::cout << "end: " << i << " cost " << best_cost << std::endl << std::flush;
        // std::cout << "no_improve_limit " << no_improve_limit << std::endl << std::flush;

        if (!best_ptr) [[unlikely]] {throw std::runtime_error("No solutions found");}
        return std::move(*best_ptr);
    }
};




py::tuple ant_colony_fromNumpy(
    const py::array_t<std::int8_t>& matrix,
    const py::array_t<std::int8_t>& flat_demons,
    const py::array_t<std::int8_t>& demons_costs,
    int buffer_size,
    int n,
    int num_demons,
    const py::array_t<int>& demons_lengths,
    int max_demon_len,
    const py::array_t<double>& heuristic,
    double alpha,
    double beta,
    double evaporation,
    double q,
    unsigned int seed,
    int n_ants,
    int n_iterations,
    int stagnant_limit
    ) {

    // I genuinely hate how complicated that comparing to brute-force args casting

    // matrix
    auto matrix_buf = matrix.request();
    if (matrix_buf.ndim != 2 || matrix_buf.shape[0] != n || matrix_buf.shape[1] != n)
        throw std::runtime_error("Matrix must be 2d with shape (n, n)");
    std::vector<int> matrix_cpp(n * n);
    auto m_ptr = static_cast<std::int8_t*>(matrix_buf.ptr);
    for (int i = 0; i < matrix_buf.size; ++i)
        matrix_cpp[i] = static_cast<int>(static_cast<unsigned char>(m_ptr[i]));

    // flat_demons
    auto demons_buf = flat_demons.request();
    if (demons_buf.size != num_demons * max_demon_len)
        throw std::runtime_error("Invalid flat_demons size");
    std::vector<int> demons_cpp(demons_buf.size);
    auto d_ptr = static_cast<std::int8_t*>(demons_buf.ptr);
    for (int i = 0; i < demons_buf.size; ++i) {
        demons_cpp[i] = static_cast<int>(static_cast<unsigned char>(d_ptr[i]));
    }

    // demons_costs
    auto costs_buf = demons_costs.request();
    if (costs_buf.size != num_demons)
        throw std::runtime_error("Invalid demons_costs size");
    std::vector<int> costs_cpp(num_demons);
    auto c_ptr = static_cast<std::int8_t*>(costs_buf.ptr);
    for (int i = 0; i < num_demons; ++i)
        costs_cpp[i] = static_cast<int>(static_cast<unsigned char>(c_ptr[i]));

    // demon_lengths
    auto len_buf = demons_lengths.request();
    if (len_buf.size != num_demons)
        throw std::runtime_error("Invalid demons_lengths size");
    std::vector lengths_cpp(static_cast<int*>(len_buf.ptr),
                               static_cast<int*>(len_buf.ptr) + num_demons);

    // heuristic
    // I could move generating it here, but numpy exec time already negligible for each run so whatever
    auto h_buf = heuristic.request();
    if (h_buf.size != n * n)
        throw std::runtime_error("Invalid heuristic size");
    auto *h_ptr = static_cast<double *>(h_buf.ptr);

    Solver solver(
        matrix_cpp.data(),
        demons_cpp.data(), lengths_cpp.data(),
        costs_cpp.data(),
        num_demons, buffer_size, n, max_demon_len,
        h_ptr, alpha, beta, evaporation, q, seed);

    SolCandidate best = solver.run_ants(n_ants, n_iterations, stagnant_limit);

    // path to ndarray
    std::vector<py::ssize_t> path_shape = {
        static_cast<py::ssize_t>(best.length),
        2
    };
    py::array_t<std::int8_t> path_array(path_shape);
    auto path = path_array.mutable_unchecked<2>();
    for (size_t i = 0; i < best.length; ++i) {
        path(i, 0) = static_cast<std::int8_t>(best.path[i].first);
        path(i, 1) = static_cast<std::int8_t>(best.path[i].second);
    }

    // buffer_seq to ndarray
    py::array_t<std::int8_t> buffer_array(buffer_size);
    auto buffer = buffer_array.mutable_unchecked<1>();
    for (int i = 0; i < buffer_size; ++i) {
        buffer(i) = static_cast<std::int8_t>(best.buffer_seq[i]);
    }

    return py::make_tuple(path_array, best.cost, best.length, buffer_array);
}



PYBIND11_MODULE(ant_colony_cpp, m) {
    m.doc() = "Solve breach protol matrix using ant-colony optimization algorythm, require padding and pre-processing";

    m.def("ant_colony", &ant_colony_fromNumpy,
        py::arg("matrix"),
        py::arg("flat_demons"),
        py::arg("demons_costs"),
        py::arg("buffer_size"),
        py::arg("n"),
        py::arg("num_demons"),
        py::arg("demons_lengths"),
        py::arg("max_demon_len"),
        py::arg("heuristic"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("evaporation"),
        py::arg("q"),
        py::arg("seed"),
        py::arg("n_ants"),
        py::arg("n_iterations"),
        py::arg("stagnant_limit")
    );
}




