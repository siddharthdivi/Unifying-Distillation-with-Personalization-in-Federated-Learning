#!/bin/bash

#######################################
## Result collection for DS-1.
#######################################

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.02 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 1 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.02 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 1 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.02 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 1 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.02 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 1 > pFedMe.out &
wait

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.02 --personal_learning_rate 0.0025 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 1 > pFedMe.out &


#######################################
## Result collection for DS-2.
#######################################

# Collect the results for DS-2 of MNIST.
nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0075 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 2 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0075 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 2 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0075 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 2 > pFedMe.out &
wait

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0075 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 2 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 128 --learning_rate 0.01 --personal_learning_rate 0.0075 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 2 > pFedMe.out &


#######################################
## Result collection for DS-3.
#######################################
nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 0 --dataSplitStrategy 3 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 1 --dataSplitStrategy 3 --gpu 0 > pFedMe.out &
wait

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 2 --dataSplitStrategy 3 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 3 --dataSplitStrategy 3 --gpu 0 > pFedMe.out &

nohup python3 -u main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.05 --beta 2 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 1 --round_num 4 --dataSplitStrategy 3 --gpu 0 > pFedMe.out &