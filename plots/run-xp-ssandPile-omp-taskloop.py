#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_taskloop -- col=schedule row=label

easypapOptions = {
    "-k": ["ssandPile"],
    "-v": ["omp"],
    "-s": [512],
    "-of": ["ssandPile_taskloop.csv"],
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

print("Recommended plot:")
print(" plots/easyplot.py -if ssandPile_taskloop.csv -v omp -- col=schedule")