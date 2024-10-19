#!/usr/bin/env python3
from expTools import *

# Recommanded plot :
# ./plots/easyplot.py -if heat-mandel.csv --plottype heatmap -heatx tilew -heaty tileh -v omp_tiled -- row=schedule aspect=1.8

easypapOptions = {
    "-k": ["ssandPile"],
    "-v": ["omp_taskloop"],
    "-s": [1024],
    "-th": [2**i for i in range(3, 11)],
    "-tw": [2**i for i in range(3, 11)],
    "-wt": ["opt"],
    "-of": ["heat-omp-taskloop-1024.csv"],
    "-i" : [1024]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_NUM_THREADS": [44],
}

execute("./run ", ompICV, easypapOptions, verbose=True, easyPath=".", nbruns=2)

# Lancement de la version seq avec le nombre de thread impose a 1

ompICV = {"OMP_NUM_THREADS": [1]}

del easypapOptions["-th"]
del easypapOptions["-tw"]
easypapOptions["-v"] = ["seq"]

execute("./run ", ompICV, easypapOptions, verbose=False, easyPath=".")


print("Recommended plot:")
print(" ./plots/easyplot.py -if heat-omp-taskloop-1024.csv --plottype heatmap", 
      "-heatx tilew -heaty tileh -v omp_tiled -- aspect=1.8")
