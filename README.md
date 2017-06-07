# ComEmb
This repository contain the code of the following paper:

"From Node Embedding To Community Embedding".  
Vincent W. Zheng, Sandro Cavallari, Hongyun Cai, Kevin Chen-Chuan Chang, Erik Cambria. 
In CoRR, 2016. 
https://arxiv.org/abs/1610.09950.

To use this code, please kindly cite our paper:
@article{DBLP:journals/corr/ZhengCCCC16, 
author    = {Vincent W. Zheng and  
               Sandro Cavallari and 
               HongYun Cai and 
               Kevin Chen{-}Chuan Chang and 
               Erik Cambria}, 
  title     = {From Node Embedding To Community Embedding}, 
  journal   = {CoRR}, 
  volume    = {abs/1610.09950}, 
  year      = {2016}
}

ComEmb learn a continuous graph representation taking in consideration first and second order proximity as well as the communities in the graph. 


All the code is implemented for python 3.6 whereas for speed up the computation, the core algorithm is implemented in cython (or python if you are not abel to compile the cython code).
To correctly execute the project you need the following packages:
 - sklearn 0.18.1
 - scipy 0.19.0
 - numpy 1.12.1
 - psutil
 - networkx 1.11
 - cython 0.25.2
 
 
# Compiling cython code
 In order to execute the cython code you have to compile it. The core algorithm is written in the file utils/training_sdg_inner.pyx or utils/embedding.py.
 To correctly compile the cython code you have to first execute the cython_utils.py file. 
 This is done with the following terminal command:
 
  - python cython_utils.py build_ext --inplace
  
 This command will generate the files training_sdg_inner.c and training_sdg_inner.so . Please make sure that both the files are in the utils folder 

# Input
The supported input format is an edgelist:

        source_node \t destination_node


# Output
The output file has n line, one for each node.
Each line follow the given template:

        node_id \t dim_1 \s dim_2 ... dim_d
 
where dim_1, ... , dim_d is the *d*-dimensional representation learned
