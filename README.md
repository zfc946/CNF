# Constrained Neural Fields

This repository is the official implementation of the paper *Neural Fields with Hard Constraints of Arbitrary Differential Order*, NeurIPS 2023.

### [Project Page](https://zfc946.github.io/CNF.github.io/) | [Paper](https://arxiv.org/abs/2306.08943)

Still training a neural field with a regression loss and hoping for it to overfit? We **enforce the regression loss to be zero as a hard constraint** in the formulation of a neural field.

## Getting Started

### Prerequisites

* Python 3.x

### Dependencies
```
pip install -r requirements.txt
```

## Demos

### Fermat's Principle

```
python fit_Fermat.py
```
![](https://zfc946.github.io/CNF.github.io/static/images/fermat.svg)

### Learning Material Appearance
First, download the MERL dataset from https://www.merl.com/brdf/, replace `PATH/TO/MERL/DATASET` at line 203 of `fit_brdf.py` with the path to the MERL dataset, then run
```
python fit_brdf.py
```
![](https://zfc946.github.io/CNF.github.io/static/images/brdf.png)

### Interpolatory Shape Reconstruction

#### Exact Normal Reconstruction 
```
python fit_exact_normal.py
```
#### Large Scale Reconstruction (Sparse Solver)
```
python fit_pointcloud_3d.py
```
![](https://zfc946.github.io/CNF.github.io/static/images/surface.png)

### Self-Tuning PDE Solver
```
python fit_advection.py
```
![](https://zfc946.github.io/CNF.github.io/static/images/advection_vis.svg)
