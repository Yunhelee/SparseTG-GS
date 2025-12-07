# SparseAG_GS

**<p align="center">An efficient and sparse adversarial test case generation method for CV(computer Vision) software</p>**

<p align="center">Code release and supplementary materials for：</p>

**<p align="center">"SparseAG-GS: Adversarial Test Case Generation via Sparse Perturbation Group"</p>**

## Datasets
- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/)

## Dependencies
The code was tested with:
- h5py                         3.8.0
- ipykernel                    6.19.2
- matplotlib                   3.7.2
- numpy                        1.25.2
- pandas                       1.5.3
- scikit-image                 0.21.0
- scipy                        1.9.3
- torch                        1.11.0
- torchvision                  0.12.0
- tqdm                         4.64.1

## Training
Training the improved AdvGAN for generating the importance matrix of perturbations.

    python train_advGAN.py

## Evaluations
1. Non-target attacks on CIFAR-10
    ```
    python SparseAG_GS_cifar.py
    ```
2. Target attacks on CIFAR-10
    ```
    python SparseAG_GS_Tarcifar.py
    ```
3. Non-target attacks on ImageNet
    ```
    python SparseAG_GS_imagenet.py
    ```
4. Target attacks on ImageNet
    ```
    python SparseAG_GS_imagenetTar.py 
    ```
5. Ablation study
    ```
    python aeCifar.py 
    ```
