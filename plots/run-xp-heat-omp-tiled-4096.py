#!/usr/bin/env python3
from expTools import *

# Recommanded plot :
# ./plots/easyplot.py -if heat-mandel.csv --plottype heatmap -heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8

easypapOptions = {
    "-k": ["ssandPile"],
    "-v": ["omp_tiled"],
    "-s": [4096],
    "-th": [2**i for i in range(3, 13)],
    "-tw": [2**i for i in range(3, 13)],
    "-wt": ["opt"],
    "-of": ["heat-omp-tiled-4096.csv"],
    "-i" : [64]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["static","static,1", "dynamic"],
    "OMP_NUM_THREADS": [44],
}

execute("./run ", ompICV, easypapOptions, verbose=True, easyPath=".")

# Lancement de la version seq avec le nombre de thread impose a 1

ompICV = {"OMP_NUM_THREADS": [1]}

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")


print("Recommended plot:")
print(" ./plots/easyplot.py -if heat-omp-tiled-4096.csv --plottype heatmap", 
      "-heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8")
