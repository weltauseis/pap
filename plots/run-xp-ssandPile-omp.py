#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label

easypapOptions = {
    "-k": ["ssandPile"],
    "-v": ["omp"],
    "-s": [512],
    "-of": ["ssandPile_xp_omp.csv"],
    "-wt": ["opt"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1", "dynamic"],
    "OMP_NUM_THREADS": [1] + list(range(4, os.cpu_count() + 1, 4)),
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
easypapOptions = {
    "-k": ["ssandPile"],
    "-v": ["seq"],
    "-s": [512],
    "-of": ["ssandPile_xp_omp.csv"],
    "-wt": ["opt"]
}
ompICV = {"OMP_NUM_THREADS": [1]}
execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if ssandPile-xp.csv -v omp -- col=schedule")