#!/usr/bin/env python3
from expTools import *

# Recommanded plot :
# ./plots/easyplot.py -if heat-mandel.csv --plottype heatmap -heatx tilew -heaty tileh -v omp -- row=schedule aspect=1.8

easypapOptions = {
    "-k": ["asandPile"],
    "-v": ["omp"],
    "-s": [512],
    "-th": [2**i for i in range(3, 10)],
    "-tw": [2**i for i in range(3, 10)],
    "-of": ["heat-omp-async.csv"],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["dynamic", "static,1"],
    "OMP_NUM_THREADS": [os.cpu_count() // 2],
    "OMP_PLACES": ["cores"],
}

execute("./run ", ompICV, easypapOptions, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

ompICV = {"OMP_NUM_THREADS": [1]}

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")

print("Recommended plot:")
print(" ./plots/easyplot.py -if heat-omp-async.csv --plottype heatmap", 
      " -heatx tilew -heaty tileh -v omp -- row=schedule aspect=1.8")
