


## INPUT:
##   A -- binary matrix (n times n)
##   clustering -- [n] -> [K] array of cluster membership
##
## OUTPUT:
##   (p, q) of estimated probabilities
##
##

estimateProb <- function(A, clustering){

    K = max(clustering)
    n = nrow(A)

    Acopy = A
    
    p_num = 0
    p_denom = 0
    
    for (k in 1:K){

        cluster_k = clustering==k
        p_num = p_num + sum(sum(A[cluster_k, cluster_k] ))
        p_denom = p_denom + sum(cluster_k)*(sum(cluster_k)-1)

        Acopy[cluster_k, cluster_k] = 0
    }

    q_num = sum(sum(Acopy))
    q_denom = n*(n-1) - p_denom

    return(c(p_num/p_denom, q_num/q_denom))
}
        
    
