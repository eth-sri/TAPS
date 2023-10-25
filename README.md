# TAPS: Connecting Certified and Adversarial Training

This is the code repository for [TAPS](https://arxiv.org/abs/2305.04574).

## Set Up

We use python version 3.9 and pytorch 1.13.1, which can be installed in a conda environment as follows:
```bash
conda create -y --name TAPS python=3.9
conda activate TAPS
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

 To install further requirements please run 
```bash
pip install -r requirements.txt
```

### Training

To train our models for MNIST, CIFAR-10, and TinyImageNet using both TAPS and STAPS, please use the commands provided in:
```bash
./scripts/train_mnist
./scripts/train_cifar
./scripts/train_tinyimagenet
```

Before training with TinyImagenet, make sure to download the dataset by running
```bash
bash ./scripts/tinyimagenet_download.sh
```

### Certification

We combine IBP, CROWN-IBP and [MN-BaB](https://github.com/eth-sri/mn-bab) for certification. 
To set up certification, please install MN-BaB as (note that you might need a ssh key in your GitHub setting to clone the repository if you get permission denied error)
```bash
git clone --branch SABR_ready --recurse-submodules https://github.com/eth-sri/mn-bab
cd mn-bab
source setup.sh
cd ..
```
Now use the commands provided in ```./scripts/mnbab_certify``` to run certification.

## Released Models

We release all the models reported in our paper in the following [link](https://mega.nz/file/TBIBHB4R#iKOm_JukfJGauQCTdrrneIQ3xhoYvQkAuSPy87899Pw).
For every model, we include the complete training log (```monitor.json```), training arguments (```./train_args.json```), certification log (```./complete_cert.json```), and certification arguments (```./cert_args.json```) in the corresponding directories.


## Reproduce Figures and Tables

Note that all the following codes are for illustrative purpose since the processing of data directory is not necessarily aligned. If the model is not explicitly provided, it refers to the final models we released.

- Figure 1, 5, 6 and 9: ```theory_TAPS_approximation.py``` computes the estimated margin ($\max_{i \ne c} y_i - y_c$) via IBP (sound over-approximation), PGD (margin attack with 3 restarts), MILP (exact solution but slow), SABR (SOTA method, unsound), TAPS (this work, unsound but precise) over models trained with these methods. The corresponding command is included in ```./scripts/theory_TAPS_approximation``` and the results are included in theory_approximation. ```./theory_approximation/mnist/eps0.05/models``` contains the models (the target $\epsilon$ is 0.05) and ```./theory_approximation/mnist/eps0.05/lower_bound``` contains the results. ```plot_TAPS_approximation.py``` contains the plotting code.
- Table 10: ```GoF_computation.py``` computes Table 10 from the trained final models. It retrieved TAPS estimated robust accuracy from ```monitor.json``` and MN-BaB certified accuracy from ```complete_cert.json``` and then computes the difference.
- Figure 10 and 11: ```train_quality_measure.py``` loads each model and test under various splits. The corresponding command is included in ```./scripts/measure_train_quality```.

## Core Code Structure

Here we present the functionality of the core files.
- ```mix_train.py``` is the main file for training models.
- ```args_factory.py``` defines the arguments.
- ```loaders.py``` defines the pipeline to process data and return dataloaders.
- ```attacks.py``` defines the adversarial attacks.
- ```torch_model_wrapper.py``` wraps IBP, PGD, SABR, TAPS and STAPS implementation into a consistent format and is called by other files.
- ```regularization.py``` defines $L_1$ and fast regularization used in IBP.
- ```utils.py``` contains utility functions, including the definition of $\epsilon$-scheduler.
- ```PARC_networks.py``` defines the network structure. The most important one is ```cnn_7layer_bn```, which is also used by SABR and fast-IBP paper.


License and Copyright
---------------------

* Copyright (c) 2023 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0)