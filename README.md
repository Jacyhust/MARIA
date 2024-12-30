# The Source Code for MARIA (Submitted to SIGMOD 2025 Round 4)
-----------------------------------------------------------------------------------------------------------------
## Introduction
This is a source code for the algorithm described in the paper **MARIA: Towards Efficient Approximate Maximum Inner Product Search on Large-Scale Vector Datasets**. We call it as **maria** project.
**maria** project is written by **C++17** and can be complied by **g++9.5.0** in **Linux** and **MSVC** in **Windows**. It adopt `openMP` for parallelism.

### Usage
#### Windows
We can use **Visual Studio 2019 or higher** to build the project with importing he files `./test/maria.cpp`.

#### Linux
```bash
make
./maria toy
```

### Dataset Format

In our project, the format of the input file (such as `audio.data_new`, which is in `float` data type) is the same as that in [LSHBOX](https://github.com/RSIA-LIESMARS-WHU/LSHBOX). It is a binary file, which is organized as the following format:

>{Bytes of the data type (int)} {The size of the vectors (int)} {The dimension of the vectors (int)} {All of the binary vector, arranged in turn (float)}


For your application, you should also transform your dataset into this binary format, then rename it as `[datasetName].data_new` and put it in the directory `./dataset`.

A sample dataset `toy.data_new` has been put in the directory `./dataset`.
