##########################################################################################################

		UNIFYING DISTILLATION WITH PERSONALIZATION IN FEDERATED LEARNING

##########################################################################################################

# Unifying-Distillation-with-Personalization-in-Federated-Learning
Repository that contains the code for the paper titled, 'Unifying Distillation with Personalization in Federated Learning'.

NOTE: The data required to run the experiments is available at https://tinyurl.com/1hp9ywfa.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SECTION-1: SETTING UP THE ENVIRONMENT.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

* Instructions to replicate the experiments in the paper.

* Organisation of this directory.

	- Unifying Distillation with Personalization/ in Federated Learning
		|
		|
		|
		| - Personalized-Federated-Learning.tar.gz
		|
		|
		| - Personalised-Federated-Learning-data.tar.gz
		|
		|
		| - pFedMe.tar.gz
		|
		|
		| - README.txt (this file.)

* All the folders with the associated code and the data have been compressed into tar files.

* First step is to untar all the tar files. 

* Once the files have been decompressed, next create a new folder in the same level as the other folders and name it 'Personalised-Federated-Learning-results' (this is the folder where all the results will be written to).

* Inside 'Personalised-Federated-Learning-results', create the following folders with the below given organisational structure: config/, EpochResults/, log/, results/, state_dict/.
	
	-Personalized-Federated-Learning-results/
		|
		|
		| - config/
		|
		|
		| - EpochResults/
		|
		|
		| - log/
		|
		|
		| - results/
		|
		|
		| - state_dict/

* Then, go to the folder 'Personalized-Federated-Learning/Personalized_Federated/code/cifar/' and read the file named 'experiments_Replication.txt' to generate the experimental results on CIFAR-10.

* After this, go to the folder 'Personalized-Federated-Learning/Personalized_Federated/code/mnist/' and read the file named 'experiments_Replication.txt' to generate the experimental results on MNIST.

* Once the previous two steps are complete, then go to the folder 'pFedMe'.

	<> If the results/ directory is not already existing, then create a new results/ directory.

	<> Then, read the experiments_Replication.txt in pFedMe/ to understand how to generate the experimental results for CIFAR-10 and MNIST for pFedMe and Per-FedAvg.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SECTION-2: CONDUCTING EXPERIMENTS ON CIFAR-10 AND MNIST FOR PersFL-KD, FedAvg, FedPer, PersFL-GD

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

* Running the experiments on CIFAR-10.
	<> cd to ~/Personalized-Federated-Learning/Personalized_Federated/code/cifar/.
	<> Run the following commands on the terminal:

		(1) To generate the results for the FedAvg model on CIFAR-10
			nohup bash FedAvg.sh > FedAvg.out &
    
		(2) To generate the results for the FedPer model on CIFAR-10
    			nohup bash FedPer.sh > FedPer.out &
    
		(3) To generate the results for the PersFL model on CIFAR-10
    			nohup bash PersFL-KD.sh > PersFL-KD.out &
    
		(4) To generate the results for the variant of PersFL model on CIFAR-10
    			nohup bash PersFL-GD.sh > PersFL-GD.out &

		(5) To generate the results for Table 4: Opt teachers vs FedAvg model as teacher model for distillation experiment.
    			nohup bash PersFL-KD-GlobInit.sh > PersFL-KD-GlobInit.out &

* Running the experiments on MNIST.
	<> cd to ~/Personalized-Federated-Learning/Personalized_Federated/code/mnist/.
	<> Run the following commands on the terminal:

		(1) To generate the results for the FedAvg model on MNIST
    			nohup bash FedAvg.sh > FedAvg.out &
    
		(2) To generate the results for the FedPer model on MNIST
    			nohup bash FedPer.sh > FedPer.out &
    
		(3) To generate the results for the PersFL model on MNIST
    			nohup bash PersFL-KD.sh > PersFL-KD.out &
    
		(4) To generate the results for the variant of PersFL model on MNIST
    			nohup bash PersFL-GD.sh > PersFL-GD.out &

		(5) To generate the results for Table 5: Opt teachers vs FedAvg model as teacher model for distillation experiment.
    			nohup bash PersFL-KD-GlobInit.sh > PersFL-KD-GlobInit.out &


* Once these experiments are done running, the results for these experiments will be stored in ./EpochResults/ folder as pickle (pkl) files.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	SUB-SECTION-2.1: ANALYSIS OF THE EXPERIMENTAL RESULTS ON CIFAR-10 AND MNIST FOR PersFL-KD, FedAvg, FedPer, PersFL-GD

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
* For an analysis of these results, please take a look at the Jupyter notebooks under the folder: '~/Personalized-Federated-Learning/Personalized_Federated/code/cifar/experiments_Replication/' for CIFAR-10 and under '~/Personalized-Federated-Learning/Personalized_Federated/code/cifar/experiments_Replication/' for MNIST.



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

SECTION-3: CONDUCTING EXPERIMENTS ON CIFAR-10 AND MNIST FOR pFedMe AND Per-FedAvg

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

* Running the experiments on CIFAR-10 & MNIST.
	<> cd to ~/pFedMe/.
	<> Run the following commands on the terminal:

		(1) To generate the results for the pFedMe model on CIFAR-10
    			nohup bash pFedMe_CIFAR-10.sh > pFedMe_CIFAR-10.out &
    
		(2) To generate the results for the Per-FedAvg model on CIFAR-10
    			nohup bash PerFed_CIFAR-10.sh > PerFed_CIFAR-10.out &
    
		(3) To generate the results for the pFedMe model on MNIST
			nohup bash pFedMe_MNIST.sh > pFedMe_MNIST.out &
    
		(4) To generate the results for the Per-FedAvg model on MNIST
    			nohup bash PerFed_MNIST.sh > PerFed_MNIST.out &


* Once these experiments are done running, the results for these experiments will be stored in ./results/ folder.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	SUB-SECTION-3.1: ANALYSIS OF THE EXPERIMENTAL RESULTS ON CIFAR-10 AND MNIST FOR pFedMe AND Per-FedAvg

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

* For an analysis of these results, please take a look at the Jupyter notebooks under the folder: '~/pFedMe/experiments_Replication/'.


-----
NOTE:
-----

* We conduct our experiments on top of the following publicly available code-bases:
	<> Federated Learning with Personalization Layers: https://bit.ly/35dKebE
	<> Federated Adaptation (to generate the Data-split strategy 2): https://github.com/ebagdasa/federated_adaptation
	<> pFedMe (Personalized Federated Learning with Moreau Envelopes): https://github.com/CharlieDinh/pFedMe


----------------------------------------------------------------------------------------------------------------------[EOF]----------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------[EOF]----------------------------------------------------------------------------------------------------------------------