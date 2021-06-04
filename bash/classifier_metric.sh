#!/bin/bash

declare -a model_Nh=("AllParametersLongRUNExMC_MNIST_NS10000_Nh200_lr0.01_l20.0_NGibbs10000.h5" "AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5" "AllParametersLongRUNExMC_MNIST_NS10000_Nh1000_lr0.01_l20.0_NGibbs10000.h5" "AllParametersLongRUNExMC_MNIST_NS10000_Nh2000_lr0.01_l20.0_NGibbs10000.h5")
ngibbs_Nh=(10000 10000 10000 10000)

declare -a model_Ns=("AllParameters_LongRUNExMC_nMB500_NS1000_TEMP_0_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5" "AllParameters_LongRUNExMC_nMB500_NS5000_TEMP_0_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5" "AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5" "AllParameters_LongRUNExMC_nMB500_NS20000_TEMP_0_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5" "AllParameters_LongRUNExMC_nMB2000_NS40000_TEMP_0_MNIST_Nh500_lr0.01_l20.0_NGibbs100.h5" "AllParameters_LongRUNExMC_nMB500_NS50000_TEMP_0_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5")
ngibbs_Ns=(10000 10000 10000 10000 100 10000)

declare -a model_Ngibbs=("AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs10.h5" "AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100.h5" "AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs1000.h5" "AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs10000.h5")
ngibbs_Ngibbs=(10 100 1000 10000)
train_time_Ngibbs=(92998 98304 98304 92998)

for i in ${!model_Nh[@]}
do
    ipython3 ../src/gen_data_metric_classifier.py ${model_Nh[$i]} ${ngibbs_Nh[$i]} 92998
done

for i in ${!model_Ns[@]}
do
    ipython3 ../src/gen_data_metric_classifier.py ${model_Ns[$i]} ${ngibbs_Ns[$i]} 92998
done

for i in ${!model_Ngibbs[@]}
do
    ipython3 ../src/gen_data_metric_classifier.py ${model_Ngibbs[$i]} ${ngibbs_Ngibbs[$i]} ${train_time_Ngibbs[$i]} 92998
done
