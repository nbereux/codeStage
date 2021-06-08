#!/bin/bash

mode=0
var_set=1
reg=100
gen_mult=0
lr=0.1
Ngibbs=50
Nmb=100
Nh=500
data_gen=0
ep_max=1000
Nh=500
data_gen=5

ipython3 ../src/run_GBRBM.py -- $mode $var_set $reg $gen_mult --lr=$lr --Ngibbs=$Ngibbs --Nmb=$Nmb --ep_max=$ep_max --Nh=$Nh --data_gen=$data_gen --var 1 1 1
