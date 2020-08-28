
source("main.R")
source("spectral.R")
source("estimateProb.R")
source("metrics.R")

library(RSpectra)
library(rmutil)

ntrials = 100
L_ls = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
#L_ls = c(2, 3, 4, 5, 6, 8)


ham_mat = matrix(0, ntrials, length(L_ls))

K = 3
n = 2100

p = .3
q = .27

Emat = matrix(q, n, n)
Dmat = diag(1, K) %x% matrix(p - q, n/K, n/K) 
Emat = Emat + Dmat

for (it in 1:ntrials){
    
    tmp = rbinom(n=n*n, size=1, prob=matrix(Emat, n*n, 1))
    A = matrix(as.numeric(tmp), n, n)
    diag(A) = 0
    Abin = A

    mask_mat = diag(1, K) %x% matrix(1, n/K, n/K)

    nne1 = sum(sum(A[mask_mat == 1 & A == 1] ))
    nne2 = sum(sum(A)) - nne1

    A[mask_mat == 1 & A == 1] = rnorm(nne1, 0.3, .8)
    A[mask_mat == 0 & A == 1] = rnorm(nne2, 0, 1)

    true_cluster = matrix(1:K, K, 1) %x% matrix(1, n/K, 1)

    for (il in 1:length(L_ls)){
       
        L = L_ls[il]
        Phi = function(x) { plaplace(x, m=0, s=2) }
        res = wsbmCluster(A, K, L, Phi)
        ham = hammingDist(true_cluster, res$clustering, K)/n
        ham_mat[it, il] = ham

        ham2 = hammingDist(true_cluster, res$spectral, K)/n
        
        print(paste(it, L, round(ham, 4), round(res$discrep, 4), round(ham2, 4)))
        
    }

    save(ham_mat, file="tmp.RData")
    
}
