1. Connect to Cyclecloud VM shell using bastion (not the web app)
2. cyclecloud initialize
3. cyclecloud start_cluster ccsw
4. cyclecloud add_nodes ccsw --count 2 --template gpu
5. cyclecloud show_nodes --cluster ccsw 
	**Keep checking to check to see that nodes are "started". Capture the Name of the nodes, e.g. gpu-1, gpu-2**
	**Look at the GPU scale set in the resource group and see that instances are starting**
8. Connect to login node instance via bastion - this will open a second browser window
	** The login node can be used for preprocessing, or other post cluster work. It is equipped with SLURM so that you can submit jobs from here**
9. run slogin <GPU node name>, e.g. slogin gpu-1
	**You are now on a GPU cluster node**
	**Run the NCCL tests**
	**Type exit to return to the login node**
	**whatever you install on the login node or scheduler will be installed on all nodes in the cluster. Think of the cluster as one entity. The scheduler, login node and compute nodes (htc, hpc, gpu) are all part of the cluster.**
10. ** Jobs on a SLURM cluster are submitted as SLURM jobs via bash scripts.** 
11. On the browser window with the cyclecloud VM, run the following commands to spin the nodes down:
	**cyclecloud terminate_node ccsw gpu-1**
   	**cyclecloud terminate_node ccsw gpu-2**
12. On the login node:
   	git clone https://github.com/jhajduk-microsoft/AI_on_Infra_Lab.git
   	cd AI_on_Infra_Lab
   	**Look at the Python script. It will bechmark a pretrained model and run inference. The input will be a movie review and the model will predict positive or negative sentiment**
   	**Look at submit_job.sh and note the SBATCH parameters such as the partition, output files, and time limit. Note the prerequisites. Feel free to submit your own sample text. Replace the sample text provided with your own**
   	sbatch -p gpu submit_job.sh
13. You will see a message stating the job number
14. Continue to monitor the job with squeue until no jobs are in the list.
15. Once the job is complete, ls your directory
16  cat llm_benchmark.out, the last lines should show the benchmarking and the sentiment prediction based on the review that you entered in the SLURM file (positive sentiment or negative sentiment)
