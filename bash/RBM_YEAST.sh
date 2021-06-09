#!/bin/bash

mode=0
var_set=1
reg=100
gen_mult=0
lr=0.01
Ngibbs=10
Nmb=50
ep_max=10000
Nh=100
data_gen=2

ipython3 ../src/run_GBRBM.py -- $mode $var_set $reg $gen_mult --lr=$lr --Ngibbs=$Ngibbs --Nmb=$Nmb --ep_max=$ep_max --Nh=$Nh --data_gen=$data_gen --var 0.5 0.5 0.5
