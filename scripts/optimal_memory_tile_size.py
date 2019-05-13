# Little convenience script to compute the optimal memory tile size within a
# given BRAM budget.

import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("data_size_bits")
argparser.add_argument("parallelism_n")
argparser.add_argument("parallelism_m")
argparser.add_argument("bram_width_bits")
argparser.add_argument("bram_depth")
argparser.add_argument("bram_count")
argparser.add_argument("size_n")
argparser.add_argument("size_m")
args = argparser.parse_args()

sof = int(args.data_size_bits)
pn = int(args.parallelism_n)
pm = int(args.parallelism_m)
wb = int(args.bram_width_bits)
sb = int(args.bram_depth)
nb = int(args.bram_count)
n = int(args.size_n)
m = int(args.size_m)

bram_width = pn * np.ceil((pm * sof) / wb)

if bram_width > nb:
    raise ValueError("Not enough BRAM to saturate compute")

area = sb * np.floor(nb / bram_width) * bram_width
root = area**0.5

tn = np.floor(root / pn) * pn
tm = np.floor(root / pm) * pm
while True:
    update = tm + pm
    if update * tn > area:
        break
    tm = update

candidate = (tn, tm)

tn = np.ceil(root / pn) * pn
tm = np.ceil(root / pm) * pm
while tn * tm > area:
    tm -= pm

if (abs(tn - tm) > abs(candidate[0] - candidate[1]) or tn * tm / (pn * pm) < tn):
    tn, tm = candidate

print("Tile sizes: {}x{}".format(int(tn), int(tm)))
print("Matrix sizes: {}xKx{}".format(int(tn * np.ceil(n / tn)),
                                     int(tm * np.ceil(m / tm))))
