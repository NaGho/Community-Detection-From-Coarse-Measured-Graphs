

## Entry point
##
## INPUT: A, a matrix of real numbers, sparse format
##        K, an integer
##        L, an integer, discretization level
##        Phi, transformation function, Real to Real
##
## OUTPUT: a clustering, a single vector [n] -> [K]

wsbmCluster <- function(A, K, L, Phi, P_o=NULL, Q_o=NULL, P0_o=NULL, Q0_o=NULL ){

    REG_CONST = .5
    
    n = nrow(A)
    
    ## apply transformation
    A_bounded = A
    A_bounded[A != 0] = Phi(A_bounded[A != 0])

    
    ## Discretize and add noise
    ##   1. Divide [0, 1] into L evenly spaced bins
    ##   2. treat A as a large vector and split A into sub matrices
    ##   Result is a list of 0/1 matrices
    Alabel = matrix(0, n, n)
    bin_width = 1/L 
    for (l in 1:L){
        Alabel[ A_bounded > (l-1)*bin_width & A_bounded <= l*bin_width ] = l
    }

    ## add noise
    tmp = rbinom(n=n*n, size=1, prob=REG_CONST*(L+1)/n)
    A_noise_mask = matrix(as.numeric(tmp), n, n)
    nne = sum(sum(A_noise_mask))
    Alabel[A_noise_mask == 1] = sample(0:L, nne, replace=TRUE)
    
    all_Al = list()
    for (l in 1:(L+1)){
        tmp = matrix(0, n, n)
        tmp[Alabel == (l-1)] = 1
        all_Al[[l]] = tmp
    }


    
    ## Apply spectral clustering to all 0/1 matrices
    ##    returns a vector [n] -> [K]
    discrep_ls = rep(0, L+1)
    best_discrep = 0
    best_clustering = rep(0, n)
    for (l in 1:(L+1)){
        cur_clustering = spectralCluster(all_Al[[l]], K)
        tmp = estimateProb(all_Al[[l]], cur_clustering)
        
        discrep_ls[l] = abs(sqrt(tmp[1]) - sqrt(tmp[2]))
        if (discrep_ls[l] > best_discrep){
            best_discrep = discrep_ls[l]
            best_clustering = cur_clustering
        }
    }

    ## Estimate all P_l's and Q_l's
    P_hats = rep(0, L+1)
    Q_hats = rep(0, L+1)
    for (l in 1:(L+1)){
        tmp = estimateProb(all_Al[[l]], best_clustering)
        P_hats[l] = tmp[1]
        Q_hats[l] = tmp[2]
    }

    
    ## refinement
    out_clustering = rep(0, n)
    oracle = rep(0, n)
    for (i in 1:n){

        cur_vec = rep(0, n)
        cur_vec_o = rep(0, n)
        for (l in 1:(L+1)){
            cur_Al = all_Al[[l]]
            cur_vec[cur_Al[i, ] == 1] = log(P_hats[l]) - log(Q_hats[l])
        }

        scores = rep(0, K)
        for (k in 1:K){
            scores[k] = sum(cur_vec[best_clustering == k])
        }
        out_clustering[i] = which.max(scores)

        
        if (!is.null(P_o)){
            scores_o = rep(0, K)
            cur_vec_o = log(P_o(A_bounded[i, ])) - log(Q_o(A_bounded[i, ]))
            cur_vec_o[A_bounded[i, ] == 0] = log(P0_o) - log(Q0_o)
            
            
            for (k in 1:K){
                scores_o[k] = sum(cur_vec_o[best_clustering == k])
            }
            oracle[i] = which.max(scores_o) 
        }
        
    }

    return(list(clustering=out_clustering, discrep=best_discrep, spectral=best_clustering,
                oracle=oracle))
}

