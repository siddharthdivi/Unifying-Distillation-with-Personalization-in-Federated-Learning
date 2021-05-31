#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, args):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                # We aren't able to load this model into the GPU memory, and also we need to be consistent in the usage 
                # of our deep models and hence we'll stick with a CNN based model for MNIST.
                # model = Net().to(device), model
                model = CNNMnist().to(device), model
            elif(dataset == "Cifar10"):
                model = CifarNet().to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else: 
                model = DNN(60,20,10).to(device), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i, args)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i, args)

        server.train()
        server.test()

    # Average data 
    if(algorithm == "PerAvg"):
        algorithm == "PerAvg_p"
    if(algorithm == "pFedMe"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times, args = args)
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times, args = args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Personalized learning rate to caculate theta aproximately using K steps")
    
    # The following argument is basically to determine how many times to repeat the experiments for. 
    parser.add_argument("--times", type=int, default=5, help="running time")
    
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    # Arguments required for us to make the read_cifa_data() more generic to read in the different data splits for each run across different datasplits.
    parser.add_argument("--total_num_users", type=int, default=10, help="The total number of users participating in the FL framework.")
    parser.add_argument("--num_overlapping_classes", type=int, default=4, help="The number of classes overlapping between the participating users.")
    parser.add_argument("--round_num", type=int, default=0, help="The current round number of the experiment being run.")
    parser.add_argument('--dataSplitStrategy', type=int, default=1, help="The datasplit strategy to follow. \
    1: DataSplit strategy from FedPer paper; \
    2: Datasplit strategy from Local Adaptation paper. \
    3: Datasplit strategy from NeurIPS 2020 paper. \
    4: Datasplit strategy from PerFed (Fallah et al.) paper.\
    ")
    
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu,
        args=args
        )
