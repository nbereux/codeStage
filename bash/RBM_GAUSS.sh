#!/bin/bash

mode = 0
var_set = 1
reg = 100
gen_mult = 0
lr = 0.01
Ngibbs = 10
Nmb = 100
ep_max = 1000
Nh = 100
data_gen = 1


ipython3 src/run_GBRBM.py -- $mode $var_set $reg $gen_mult --lr=$lr --Ngibbs=$Ngibbs --Nmb=$ --ep_max=1000 --Nh=500 --data_gen=0 --var 0.5 0.5 0.5