

## Spectral clustering
##
## INPUT: A, a matrix of 0/1
##        K, an integer, number of clusters
##

## OUTPUT: a clustering, a single vector [n] -> [K]

spectralCluster <- function(A, K){

    TRIM_CONST = 20
    NEIGHB_CONST = 10
    
    ## compute average degree
    n = nrow(A)
    deg_vec = apply(A, 1, sum)

    ## keep only nodes whose degree is not extreme
    keep_nodes = deg_vec <= TRIM_CONST * mean(deg_vec)
    Atrim = A[keep_nodes, keep_nodes]

    ntrim = nrow(Atrim)

    ## get best rank K approximation of Atrim: !! Ahat !!
    res = eigs_sym(Atrim, K)
    Ahat = res$vectors %*% diag(res$values) %*% t(res$vectors)


    ## Generate a list of K anchors
    ## get random list of potential anchor points
    ## if (ntrim <= 1000) {
    ##     rand_cands = sample(ntrim, floor(ntrim/10))
    ## } else if (ntrim <= 2000) {
    ##     rand_cands = sample(ntrim, floor(ntrim/20))
    ## } else {
    ##     rand_cands = sample(ntrim, floor(ntrim/30))
    ## }
    rand_cands = sample(ntrim, floor(ntrim^(2/3)))
    
    
    nrand = length(rand_cands)
    Arand = Ahat[rand_cands, rand_cands]

    Arand1 = Arand %x% matrix(1, nrand, 1)
    Arand2 = matrix(1, nrand, 1) %x% Arand

    D = apply(Arand1^2, 1, sum) + apply(Arand2^2, 1, sum) - 2 * apply(Arand1 * Arand2, 1, sum)
    D = matrix(D, nrand, nrand)

    neighb = matrix(0, nrand, nrand)
    neighb[D <= NEIGHB_CONST * K^2 * mean(deg_vec)/n] = 1

    ## get set of potential anchor points with large number of neighbors
    neighb_vec = apply(neighb, 1, sum)
    eligible_pts = which(neighb_vec > (1/NEIGHB_CONST) * nrand/K)

    
    S = rep(0, K)
    S[1] = which.max(apply(neighb, 1, sum))
    
    for (kk in 2:K){
        S[kk] = which.max(apply( matrix(D[S[1:(kk-1)], eligible_pts], kk-1, length(eligible_pts)),   2, min))
    }

    S_identity = rand_cands[S]
    
    ## Compute distance of every point in ntrim to S_identity
    A_S1 = Ahat[S_identity, ] %x% matrix(1, ntrim, 1)
    A_S2 = matrix(1, K, 1) %x% Ahat

    D_S = apply(A_S1^2, 1, sum) + apply(A_S2^2, 1, sum) - 2 * apply(A_S1 * A_S2, 1, sum)
    D_S = matrix(D_S, ntrim, K)

    ## Get final clustering
    cluster = sample(1:K, size=n, replace=TRUE)
    cluster[keep_nodes] = apply(D_S, 1, which.min)

    return(cluster)
}
    
        
        

    

    
    
    
