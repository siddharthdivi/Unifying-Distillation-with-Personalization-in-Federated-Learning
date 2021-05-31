#!/bin/bash

##########################################################
# CIFAR-10 Dataset
##########################################################

##########################################################
# DATASPLIT - 1
##########################################################
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.001 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 1 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.001 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 1 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.001 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 1 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.001 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 1 --gpu 0 > perFed.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.001 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 1 --gpu 0 > perFed.out &


##########################################################
# DATASPLIT - 2
##########################################################
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 2 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 2 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 2 --gpu 0 > perFed.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 2 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 2 --gpu 0 > perFed.out &


##########################################################
# DATASPLIT - 3
##########################################################
nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 1000 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 3 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 1000 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 3 --gpu 0 > perFed.out &
wait

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 1000 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 3 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 1000 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 3 --gpu 0 > perFed.out &

nohup python3 -u main.py --dataset Cifar10 --model cnn --batch_size 128 --learning_rate 0.01 --beta 0.0025  --num_global_iters 1000 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 3 --gpu 0 > perFed.out &