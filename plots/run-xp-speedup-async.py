#!/usr/bin/env python3
from expTools import *

easypapOptions = {
    "-k": ["asandPile"],
    "-v": ["omp"],
    "-s": [512],
    "-of": ["speedup-async.csv"],
    "-tw": ["512"],
    "-th": ["8"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static", "static,1", "dynamic"],
    "OMP_NUM_THREADS": list(range(4, os.cpu_count() + 1, 4)),
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1
ompICV = {"OMP_NUM_THREADS": [1]}

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]

execute("./run ", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Recommended plot:")
print(" plots/easyplot.py -if speedup-async.csv -v omp -- col=schedule")