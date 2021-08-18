# Graph Community Detection From Coarse Measurements
## Recovery Conditions for the Coarsened Weighted Stochastic Block Model

This repository contains the code to reproduce the results in our paper [1]. we evaluate the error behavior of the community recovery from synthetically generated coarse measured graphs. 

We compare the theoretical error bounds derived in Sec.  of the paper, with state-of-the-art community detection methods from existing works that are applied to the generated 
coarse graphs.

The methods include Modularized non-negative matrix factorization (M-NMF) [2], Speaker-listener Label Propagation Algorithm (SLPA) [3], 
Non-Negative Symmetric Encoder-Decoder (NNSED) [4], and Cluster Affiliation Model for Big Networks (BigClam) [5]. 
These methods are tried for various hyper-parameters and the ones that result in the best performance are selected.

Please use the following for citation when using the code:
    [1] Nafiseh Ghoroghchian, Gautam Dasarathy, and Stark C. Draper. ``Graph community detection from coarse measurements'',
        accepted and to be published in the proceedings of the 24th international conference on artificial intelligence and statistics (AISTATS), 2021.
when using the code in this directory.

Other references:
[2] \citep{wang2017community}
[3] \citep{xie2011slpa,xie2013overlapping}
[4] \citep{sun2017non}
[5] \citep{yang2013overlapping}
