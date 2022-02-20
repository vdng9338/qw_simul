Code for quantum walk simulations
====

This repository contains code for simulating various quantum walks on triangular
lattices.

# Requirements
The code requires the libraries Eigen and a modified version of
![matplotlib-cpp](https://github.com/lava/matplotlib-cpp) (included in
matplotlibcpp.h). To install the dependencies on Ubuntu 20.04:
```
sudo apt-get install python3-matplotlib python3-numpy python3.8-dev libeigen3-dev
```
(Later versions of Python 3 should also work.)

# Building
A Makefile is provided. Alternatively, compilation commands are included at the
beginning of each `triangular_*.cpp` source file. (Both are for Python 3.8, but
later versions of Python 3 should also work.)