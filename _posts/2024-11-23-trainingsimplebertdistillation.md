---
layout: post
title: "Training a simple BERT module on a single A10G"
date: 2024-11-23
categories: ai
author: Bruce Changlong Xu
---

This post documents the process of training a simple BERT module on a single A10G GPU. There are minor difficulties and challenges that are used to manage GPU clusters that one must familiarize themselves with in order to truly work productively with AI. Recall, that in a typical AI workflow, we have two phases - "**training**" and "**inference**".

**ssh into server** and **cloning repo**: Our first order of business, is typically to ssh into a server with access to a compute cluster that we can run our workloads on. For instance, I can run the following command in my VScode terminal "ssh b{....}@b{.....}", which allows me to ssh into my desired server. From here, we should clone the github repo that we want to run (which contains our AI model infrastructure). Remember here, we should add our dev server ssh key (public) to the list of accessible ssh keys on Github to be able to clone our desired repo. 

The next order of business is to **install all dependencies** and **setting up a clean environment**. We need to ensure compatibility with Python, Pytorch, CUDA, our NVIDIA GPU version and other dependencies in order to successfully run everything required in our script. To this end **miniconda** environments are extraordinarily useful. We should create a conda environment with our specified Python version, and subsequently install all of the dependencies in this environment. We should make sure that pip is sufficiently up to date when downloading these packages, and also that the CUDA version is supported by Pytorch. 

**Preprocessing data:** Typically there is a "preprocess.py" script that is used to preprocess the data that we need to feed into our model. We need to feed in the current arguments to this script to generate the desired preprocessed data files. Now our training file "train.py" must also take in the correct arguments to successfully run. If we receive an error like follows "RuntimeError: CUDA error: invalid device ordinal" this means that there is an error with the GPU device indices, and so we should check our GPU resource constraints with "**nvidia-smi**" to confirm that (e.g. only one GPU is available, GPU device 0). We can then set the environment variable directly from the terminal to specify the correct GPU "export CUDA_VISIBLE_DEVICES=0" and update the training command to use only one GPU "python train.py --world_size 1 --gpus 1 [other arguments...]". 

Some example of errors that we could potentially run into are "cublas runtime error" which tells us that there is an error during GPU operations during tensor multiplication. To solve this we could:

- Check tensor shapes before the operation to ensure they were compatible.
- Verify that tensors did not contain NaNs or Infs
- Reduce the batch size to alleviate potential memory issues
- Ensure that PyTorch and CUDA versions are compatible
- Attempt to use mixed-precision training and update model initialization to prevent numerical instability 

Finally, we can use the command **nvidia-smi** once more to understand the utilization of our GPU during training/inference and running of the script. To effectively monitor GPU utilization during training on a cluster, you can use several techniques. For seamless multitasking, terminal multiplexers like tmux or screen allow you to split windows or detach sessions, enabling simultaneous training and GPU monitoring with tools like nvidia-smi. Alternatively, run the training script in the background using nohup and use the terminal for monitoring or checking logs. For a cleaner approach, open a separate cbrun session dedicated to GPU monitoring. Integrating GPU utilization logging directly into your training script offers real-time insights in the output logs. Additionally, tools like gpustat or remote monitoring via SSH tunneling provide comprehensive GPU metrics, even from a distance. These strategies ensure smooth and effective GPU management without interrupting your training workflow.
