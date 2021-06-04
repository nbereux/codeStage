import torch
import matplotlib.pyplot as plt
import numpy as np

data_nh = torch.load("data/classification_error/classification_error_Nh.pt")
plt.plot(data_nh[0], data_nh[1], '-o')
plt.xlabel("Number of hidden neurons")
plt.ylabel("Missclassification rate")
plt.savefig("fig/classifier_metric/plot_Nh.png")
plt.close()

data_ns = torch.load("data/classification_error/classification_error_Ns.pt")
plt.plot(data_ns[0], data_ns[1], '-o')
plt.semilogx()
plt.xlabel("Training sample size")
plt.ylabel("Missclassification rate")
plt.savefig("fig/classifier_metric/plot_Ns.png")
plt.close()

data_ngibbs = torch.load(
    "data/classification_error/classification_error_Ngibbs.pt")
plt.plot(data_ngibbs[0], data_ngibbs[1], '-o')
plt.semilogx()
plt.xlabel("NGibbs")
plt.ylabel("Missclassification rate")
plt.savefig("fig/classifier_metric/plot_Ngibbs.png")
plt.close()

data_100 = torch.load(
    "data/classification_error/classification_error_alltime_AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100.h5.pt")

plt.xlim(left=100, right=10**4)
plt.semilogx(data_100[0], data_100[1])
plt.xlabel("Nupdate")
plt.ylabel("Missclassification rate")
plt.savefig(
    "fig/classifier_metric/plot_alltime_AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100.png")
plt.close()

newval = torch.load(
    "data/classification_error/classification_error_newtime_AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100.h5.pt")
plt.xlim(left=100, right=10**4)
plt.semilogx(newval[0], newval[1])
plt.hlines(0.5, xmin=100, xmax=10**4, color='red')
plt.xlabel("Nupdate")
plt.ylabel("Missclassification rate")
plt.savefig(
    "fig/classifier_metric/plot_newtime_AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100.png")
plt.close()

data_norm = newval[1]-0.5
data_norm = np.power(data_norm, 2)

plt.xlim(left=100, right=10**4)
plt.loglog(newval[0], data_norm)
plt.xlabel("Nupdate")
plt.ylabel("Squared difference from 0.5")
plt.savefig(
    "fig/classifier_metric/plot_newtime_AllParametersLongRUNExMC_MNIST_Nh500_lr0.01_l20.0_NGibbs100_diff.png")
plt.close()
