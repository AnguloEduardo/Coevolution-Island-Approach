import tkinter as tk
from tkinter import messagebox
from main_HyperSolver import main


def run_algorithm():
    try:
        # Fetch parameters
        number_of_tournaments = int(entry_number_of_tournaments.get())
        number_of_parents_to_select = int(entry_number_of_parents_to_select.get())
        individuals_to_exchange = int(entry_individuals_to_exchange.get())
        number_of_islands = int(entry_number_of_islands.get())
        population_size = int(entry_population_size.get())
        generations = int(entry_generations.get())
        crossover_probability = list(map(float, entry_crossover_probability.get().split()))
        mutation_probability = list(map(float, entry_mutation_probability.get().split()))
        migration_probability = float(entry_migration_probability.get())
        features = entry_features.get().split()
        heuristics = entry_heuristics.get().split()
        number_rules = int(entry_number_rules.get())
        training_split = entry_training_split.get()
        training_set = entry_training_set.get()
        lb = float(entry_lb.get())
        ub = float(entry_ub.get())
        nm = int(entry_nm.get())
        pc = int(entry_pc.get())

        # Run your algorithm
        your_algorithm_function(number_of_tournaments, number_of_parents_to_select, individuals_to_exchange,
                                number_of_islands, population_size, generations, crossover_probability,
                                mutation_probability, migration_probability, features, heuristics,
                                number_rules, training_split, training_set, lb, ub, nm, pc)

        messagebox.showinfo("Success", "The algorithm has been successfully run!")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.geometry('500x600')  # Set the size of the window

# Create fields for each parameter
entry_number_of_tournaments = tk.Entry(root)
entry_number_of_parents_to_select = tk.Entry(root)
entry_individuals_to_exchange = tk.Entry(root)
entry_number_of_islands = tk.Entry(root)
entry_population_size = tk.Entry(root)
entry_generations = tk.Entry(root)
entry_crossover_probability = tk.Entry(root)
entry_mutation_probability = tk.Entry(root)
entry_migration_probability = tk.Entry(root)
entry_features = tk.Entry(root)
entry_heuristics = tk.Entry(root)
entry_number_rules = tk.Entry(root)
entry_training_split = tk.Entry(root)
entry_training_set = tk.Entry(root)
entry_lb = tk.Entry(root)
entry_ub = tk.Entry(root)
entry_nm = tk.Entry(root)
entry_pc = tk.Entry(root)

# Layout fields
entry_number_of_tournaments.pack()
entry_number_of_parents_to_select.pack()
entry_individuals_to_exchange.pack()
entry_number_of_islands.pack()
entry_population_size.pack()
entry_generations.pack()
entry_crossover_probability.pack()
entry_mutation_probability.pack()
entry_migration_probability.pack()
entry_features.pack()
entry_heuristics.pack()
entry_number_rules.pack()
entry_training_split.pack()
entry_training_set.pack()
entry_lb.pack()
entry_ub.pack()
entry_nm.pack()
entry_pc.pack()

# Create a button to run your algorithm
run_button = tk.Button(root, text="Run Algorithm", command=run_algorithm)
run_button.pack()

root.mainloop()
