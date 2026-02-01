import torch


print("Dataset 1")
data1 = torch.load("data/processed/Sobol/rho_0.010_E_q_fixed/dataset_2500.pt", weights_only=False)
print(data1.keys())
print(data1["metadata"].keys())
print("")

print("Dataset 2")
data2 = torch.load("data/processed/Sobol/slenderness_method/dataset_2500.pt", weights_only=False)
print(data2.keys())
print(data2["metadata"].keys())
print("")

print("Dataset 3")
data3 = torch.load("data/processed/Sobol/slenderness_method/dataset_100000.pt", weights_only=False)
print(data3.keys())
print(data3["metadata"].keys())
print("")