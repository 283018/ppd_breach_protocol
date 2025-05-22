# main.py
"""
Na razie szybkie pokazanie wyników i wstępne porównanie solvera i bruteforce'a
"""
from core import *
from breach_solvers import *
from task_generator import *




def main():
    # tworzenie instancji fabryki zadań
    factory = TaskFactory(seed=123)
    # tworzenie instancji solverów
    gb_solver = get_solver('gurobi')
    bf_solver = get_solver('brute')

    n_iter = 20
    results = {}    # na razie używam zagnieżdżonego słownika, później wymyślę coś lepszego

    for i in range(n_iter):
        task = factory(-3)  # używamy trybu generującego w miarę łatwe zadania
        results[i] = {
            'task': task,
            'gurobi': {},
            'brute': {},
        }

        solution_1, time_1 = gb_solver(task)
        results[i]['gurobi']['solution'] = solution_1
        results[i]['gurobi']['time'] = time_1

        solution_2, time_2 = bf_solver(task)
        results[i]['task'] = task
        results[i]['brute']['solution'] = solution_2
        results[i]['brute']['time'] = time_2

        filled = (i + 1) * 20 // n_iter
        print(f'\rProgress: [{"#" * filled}{"-" * (20-filled)}] {(i + 1) / n_iter:.0%}', end='', flush=True)


    example_gb = results[0]['task'], results[0]['gurobi']['solution']
    example_bf = results[0]['task'], results[0]['brute']['solution']

    print()
    print("Przykładowy task z rozwiązaniem: ")
    print("\n    Gurobi: ")
    bprint(example_gb[0], example_gb[1])
    print("\n    Brute: ")
    bprint(example_bf[0], example_bf[1])
    print("\nPorównanie czasów i zdobytych punktów: ")
    print("         Czas                   Punkty")
    print("   gurobi | brute          gurobi | brute  ")
    for i in range(n_iter):
        gb = results[i]['gurobi']
        bf = results[i]['brute']
        comp_t = "<" if gb['time'] < bf['time'] else ">" if gb['time'] > bf['time'] else "="
        comp_p = "<" if gb['solution'].total_points < bf['solution'].total_points else \
            ">" if gb['solution'].total_points > bf['solution'].total_points else "="

        print(f"{gb['time']:6f}  {comp_t}  {bf['time']:6f}         "
              f"{gb['solution'].total_points:>2d}  {comp_p}  {bf['solution'].total_points:>2d}")




    print("\n\nPrzykłąd trudniejszego: ")
    task = factory(-1)
    solution_1, time_1 = gb_solver(task)
    solution_2, time_2 = bf_solver(task)
    bprint(task, solution_1, True)
    bprint(task, solution_2, True)
    print("   gurobi | brute")
    print(f"{time_1:>8.6f}  |  {time_2:>8.6f}")


    comment_on_solvers='''
Z obserwacji na razie widać że wyniki końcowe są takie same dla obu metod, natomiast znacznie różni się czas,
również ze względu na wynagrodzenie za niezużycie buffera, które nie jest uwzględniane w zwykłej metodzie
ścieżka z gurobi jest zazwyczaj bardziej optymalna, choć i daje tyle samo punktów.
    '''
    print(comment_on_solvers)








if __name__ == '__main__':
    main()