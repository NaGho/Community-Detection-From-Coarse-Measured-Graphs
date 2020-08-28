
## INPUT

hammingDist <- function(cluster_gold, cluster_red, K){

    ## compute best mapping
    tau = rep(0, K)

    for (k in 1:K){

        k_gold = cluster_gold == k
        overlaps = rep(-Inf, K)
        for (kk in 1:K){
            if (kk %in% tau)  next
            
            overlaps[kk] = sum(k_gold & cluster_red==kk)
        }

        tau[k] = which.max(overlaps)
    }

    hamming = 0
    for (k in 1:K) {
        hamming = hamming + sum(cluster_gold==k & cluster_red!=tau[k])
    }

    return(hamming)
}
    
            
