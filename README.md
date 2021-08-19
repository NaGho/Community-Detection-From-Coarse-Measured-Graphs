# Graph Community Detection From Coarse Measurements 
## Recovery Conditions for the Coarsened Weighted Stochastic Block Model

This repository contains the code to reproduce the results in our paper [1]. we evaluate the error behavior of the community recovery from synthetically generated coarse measured graphs. 

We compare the theoretical error bounds derived in Sec.  of the paper, with state-of-the-art community detection methods from existing works that are applied to the generated 
coarse graphs.

The methods include Modularized non-negative matrix factorization (M-NMF) [2], Speaker-listener Label Propagation Algorithm (SLPA) [3], 
Non-Negative Symmetric Encoder-Decoder (NNSED) [4], and Cluster Affiliation Model for Big Networks (BigClam) [5]. 
These methods are tried for various hyper-parameters and the ones that result in the best performance are selected.

![Figure 1: Visual illustration of (a) the underlying high-resolution (fine) graph, (b) the measurement (coarsening) procedure
whose result is modeled as a coarse graph, and (c) the effect of the coarsening on the community structure, whose
recovery is the objective of this paper. Some notations used in this paper, with their values realized for this figure, are
annotated.](https://github.com/NaGho/Community-Detection-From-Coarse-Measured-Graphs/blob/master/Coarsening_and_Community_Detection_System_Model.pdf)

![Figure 2-a: Community recovery error w.r.t. the coverage size.](https://github.com/NaGho/Community-Detection-From-Coarse-Measured-Graphs/blob/master/simulation%20results/synthetic_UB_Errors_wrt_Coverage%20Size_n30000_m400_K5_nu2.png)

![Figure 2-b: Community recovery error w.r.t. the measurement size.](https://github.com/NaGho/Community-Detection-From-Coarse-Measured-Graphs/blob/master/simulation%20results/synthetic_UB_Errors_wrt_MeasurementSize_n30000_m10_K5_nu2.png)

Please use the following for citation:
   > [1] Nafiseh Ghoroghchian, Gautam Dasarathy, and Stark C. Draper. "Graph community detection from coarse measurements",
        accepted and to be published in the proceedings of the 24th international conference on artificial intelligence and statistics (AISTATS), 2021.
        
when using the code in this directory.

Other references:

<font size="+0.5">
[2] Wang, Xiao, et al. "Community preserving network embedding." Thirty-first AAAI conference on artificial intelligence. 2017.  

[3] Xie, Jierui, Stephen Kelley, and Boleslaw K. Szymanski. "Overlapping community detection in networks: The state-of-the-art and comparative study." Acm computing surveys (csur) 45.4 (2013): 1-35.  

[4] Sun, Bing-Jie, et al. "A non-negative symmetric encoder-decoder approach for community detection." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.  

[5] Yang, Jaewon, and Jure Leskovec. "Overlapping community detection at scale: a nonnegative matrix factorization approach." Proceedings of the sixth ACM international conference on Web search and data mining. 2013.
</font>
