#!/bin/bash

# python3 -u main.py --dataset Mnist --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 2 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 2

##############################
# DATASPLIT - 1
# Hyper-params for CIFAR-10 DS-1 fixed.
# personal learning rate: 0.001 
# global epochs: 800
##############################
# # Num Epochs: 800
echo "Number of Epochs: 800."
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 1 --gpu 1 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 1 --gpu 1 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 1 --gpu 1  > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 1 --gpu 1  > pFedMe.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.001 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 1 --gpu 1  > pFedMe.out &


##############################
# DATASPLIT - 2
##############################
# Num Epochs: 1000
# echo "Number of Epochs: 1000."
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 1000 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 2 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 1000 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 2 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 1000 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 2  > pFedMe.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 1000 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 2  > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 1000 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 2  > pFedMe.out &


##############################
# DATASPLIT - 3 
##############################
# Num Epochs: 800
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 1 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 1 --gpu 0 > pFedMe.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 1 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 1 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 1 --gpu 0 > pFedMe.out &