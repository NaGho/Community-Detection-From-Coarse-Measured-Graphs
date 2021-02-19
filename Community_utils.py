import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import logsumexp
# import sys
import seaborn as sns
from GraphL_Utils import CoarseningParams, fine_mat_gen, MapFineToCoarse, plot_graph_density, SparsityCore, PlotCore
from GraphL_Utils import Fine2CoarseCore, FineMatCore, CoarseMatCore, sparsity_eval, Gsparsity_function
from GraphL_Utils import plot_array, FineCoarseMode, NormalDistProp, DeltaDistProp, SBMDistProp, UniformDistProp
from GraphL_Utils import MixtureDistProp, calc_all_sparsities, plot_line, plot_scatter, plot_regression
from GraphL_Utils import plot_graphScaling, plot_hist_text, plot_heat_text, graphPruning, plot_single_regression
from GraphL_Utils import arrayScaling, arrayShifting, arrayMaxForcing, arrayMinForcingScaling, select_fine_distribution
from GraphL_Utils import df_empty, aggregateList
import pandas as pd
from math import *
import itertools
from scipy.stats import norm
import networkx as nx
from cdlib import algorithms, ensemble, evaluation, classes
from cdlib.ensemble.bunch_executions import grid_execution
from cdlib.classes import NodeClustering
from datetime import datetime
# pd.set_option('display.width', 1000)
def nCr(n,r):
    f = factorial
    return f(n) / f(r) / f(n-r)


def profile2community(phi, nu):
    return

def CH_divergence(c1,c2):
    t = np.arange(0, 1.05, 0.05)[:,np.newaxis] # [1/2] # 
    c1 = c1[np.newaxis,:]
    c2 = c2[np.newaxis,:]
    return np.max(np.sum(t*c1+(1-t)*c2-(c1**t)*(c2**(1-t)), 1))
 
 
def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = np.zeros((n,), dtype=int)
        s[list(bits)] = 1
        result.append(s)
    return result

   
def generateNormalizedProfileSet(df):
    if(False):
        lst = list(itertools.product([0, 1], repeat=df['K']))
        phiN = np.array([s for s in lst if (np.sum(s)>0 and np.sum(s)<=df['nu'])])
    else:
        result = []
        for nu in np.arange(df['nu'])+1:
            result.extend(kbits(df['K'], nu))
        phiN = np.array(result)                      
    row_sums = phiN.sum(axis=1)
    return phiN / row_sums[:, np.newaxis]
    
    
    
def UB_Failure_Error_Function(df):
    if(df['nu']==1): # (False): # 
        g = int(df['m']/df['K'])
#         err_terms1 = [-np.inf]
#         err_terms2 = [-np.inf]
        I = -2*(df['r']**2)*np.log(np.sqrt((1-df['alpha']*np.log(df['n'])/df['n'])*(1-df['beta']*np.log(df['n'])/df['n'])) + \
                             np.sqrt(df['alpha']*df['beta'])*np.log(df['n'])/df['n'])
    #     I = 2*(C**2)*((alpha+beta)/2-np.sqrt(alpha*beta))*np.log(n)/n 
    #     if(n<20*L):
    #         I = -2*(C**2)*np.log(np.sqrt((1-alpha*np.log(n)/n)*(1-beta*np.log(n)/n))+np.sqrt(alpha*beta)*np.log(n)/n) 
    #     else:
    #         I = 2*(C**2)*((alpha+beta)/2-np.sqrt(alpha*beta))*np.log(n)/n
        print('gI/Klog g=', g*I/(np.log(g)))
        min_err = 1
        for mPrime in [int(np.floor(g/2))+1]: # np.arange(1, int(np.floor(g/2))+1):
            m = np.arange(1, mPrime) # np.arange(1, L+1) #
            err = 1 
            if(m.size>0):
                try: 
    #                 err_terms1 = np.log(np.min(np.stack(((np.exp(1)*g*(df['K']**2)/m)**m, df['K']**(g*df['K'])*np.ones_like(m)),1),1).astype('float64')) + (-g*m+m**2)*I
                    err = np.sum(np.min(np.stack(((np.exp(1)*g*(df['K']**2)/m)**m, df['K']**(g*df['K'])*np.ones_like(m)),1),1).astype('float64')* np.exp((-g*m+m**2)*I))
        #             if(err>1):
        #                 err = 1
                except OverflowError as error:
                    err = 1
                            
            m = np.arange(m[-1]+1 if m.size>0 else 1, g*df['K']+1)
            if(m.size>0): 
                try:
    #                 err_terms2 = np.log(np.min(np.stack(((np.exp(1)*g*(df['K']**2)/m)**m, df['K']**(g*df['K'])*np.ones_like(m)),1),1).astype('float64')) + (-2*g*m/9)*I
                    err += np.sum(np.min(np.stack(((np.exp(1)*g*(df['K']**2)/m)**m, df['K']**(g*df['K'])*np.ones_like(m)),1),1).astype('float64')* np.exp((-2*g*m/9)*I))
        #             if(err>1):
        #                 err = 1
                except OverflowError as error:
                    err = 1
    #         logErr = logsumexp(np.concatenate((err_terms1, err_terms2)))
            if(err<min_err):
                min_err = err
        Err = min_err
        df['Failure UB Error under CO-1 constraint'] = Err

    tau_tilde = 1e-2 # 1/(df['nu']+1)
    df['K_nu'] = 0
    for nu in np.arange(df['nu'])+1:
        df['K_nu'] += int(nCr(df['K'], nu))
    phiN = generateNormalizedProfileSet(df)
    U = np.zeros((df['K_nu'],df['K_nu']))
    
    epsilonMax = np.zeros((df['K_nu'],df['K_nu']))
    logDivn = np.log(df['n'])/df['n']
    normDot = np.matmul(phiN, phiN.T)
    if(True):
        U = norm.cdf((df['alpha']-df['beta'])*df['r']*np.sqrt(logDivn)
                        *np.divide(normDot-tau_tilde, np.sqrt((df['alpha']-df['beta'])*normDot + df['beta'])))
        epsilonMax = 0.7915/(
                    np.sqrt((df['alpha']*(1-df['alpha']*logDivn)*normDot+df['beta']*(1-df['beta']*logDivn)*(1-normDot))
                    *logDivn)*df['r']
                    )
    else:
        for k in np.arange(df['K_nu']):
            for kprime in np.arange(k, df['K_nu']):
                U[k,kprime] = norm.cdf(
                                (df['alpha']-df['beta'])
                                *(normDot[k,kprime]-tau_tilde)*df['r']*np.sqrt(logDivn)
                                /np.sqrt((df['alpha']-df['beta'])*normDot[k,kprime] + df['beta'])
                                )
                epsilonMax[k,kprime] = 0.7915/(
                    np.sqrt((df['alpha']*(1-df['alpha']*logDivn)*normDot[k,kprime]+df['beta']*(1-df['beta']*logDivn)*(1-normDot[k,kprime]))
                    *logDivn)*df['r']
                    )
        epsilonMax += epsilonMax.T  
        epsilonMax[np.diag_indices(epsilonMax.shape[0])] = np.diag(epsilonMax)/2
        U += U.T  
        U[np.diag_indices(U.shape[0])] = np.diag(U)/2
    
    if(True):
        prior_vec = np.array([1/(np.count_nonzero(vec)) for vec in phiN]) # 
        prior_vec = prior_vec/prior_vec.sum()
    else:
        prior_vec = 1//df['K_nu']
    ErrArray = []
    for epCoeff in [-1, 0, 1]:
        Err = 0 
        for k in np.arange(df['K_nu']):
            for kprime in np.arange(k+1, df['K_nu']):
                c1 = np.multiply(np.max([U[:,k]+epCoeff*epsilonMax[:,k],np.zeros((U.shape[1],)).T], 0), prior_vec)
                c2 = np.multiply(np.max([U[:,kprime]+epCoeff*epsilonMax[:,kprime],np.zeros((U.shape[1],)).T], 0), prior_vec)
                Err += np.exp(-df['m']*CH_divergence(c1, c2))
        if(Err>1):
            Err = 1 
#         logErr = np.sum([[L**(-CH_divergence(df['m']*U[k,:]/np.log(df['m']),df['m'])*U[kprime,:]/np.log(df['m']))\
#                           for k in np.arange(df['K_nu'])] for kprime in np.arange(df['K_nu'])])
        ErrArray.append(Err)
    df['Failure UB Error (Lowest U)'] = ErrArray[0]
    df['Failure UB Error (Highest U)'] = ErrArray[-1]
    Err = ErrArray[1]
    df['Failure UB Error'] = Err
    return Err, df


class SSBMCore():
    def __init__(self, df, measuring_core, mesh, genModule = 'networkX'):
        self.genModule = genModule
        self.numNodes = df['n']
        self.numComs = df['K']
        self.Qprobs = (df['beta']*np.ones((self.numComs,self.numComs)) + (df['alpha']-df['beta']) 
                                    * np.diag(np.ones(self.numComs,)))*np.log(df['n'])/df['n']
        
        if(measuring_core is None or measuring_core.linCoarseningMat is None):
            self.comSizes = int(np.floor(self.numNodes/self.numComs)) * np.ones((self.numComs,)).astype(int) # [50, 50, 100, 100, 150]
            self.comSizes[-1] = self.numNodes-np.sum(self.comSizes[:-1])
            if(mesh is not None):
                self.P = np.zeros((self.numComs, self.numNodes))
                self.comIndices = []
                remaining_indices = np.arange(mesh.size)
                for i in np.arange(self.numComs):
                    in_Chosen = np.random.choice(remaining_indices, size=self.comSizes[i], replace=False)
                    self.P[i,in_Chosen] = 1
                    remaining_indices = list(set(remaining_indices)-set(in_Chosen))
                    in_Chosen.sort()
                    self.comIndices.append(in_Chosen)
                
        else:
            self.normProfileMat = profileMatGen(df)
            self.P = np.zeros((self.numComs, self.numNodes))
            # Step 1: filling the determined fine nodes from the sensing matrix
            for i in np.arange(measuring_core.linCoarseningMat.shape[0]):
                B_row = measuring_core.linCoarseningMat[i,:]
                I_proto_row = np.argwhere(B_row!=0)
                splitSizes = np.cumsum(np.round(self.normProfileMat[i,:]*I_proto_row.size).astype(int))
                splitIdx = np.split(I_proto_row, splitSizes)[:-1]
                for j, idxSet in enumerate(splitIdx):
                    self.P[j, idxSet] = 1
            # Step 2: get rid of double assigned
            multAssignedNodes = np.argwhere(np.sum(self.P, axis=0)>1)[:,0]
            for i in multAssignedNodes:
                repComIdx = np.random.choice(np.argwhere(self.P[:,i]!=0)[:,0], size=1, replace=False)
                self.P[:,i] = 0
                self.P[repComIdx, i] = 1
            # Step 3: fill the rest 
            ComNonAssNodes = np.argwhere(np.sum(self.P, axis=0)==0)[:,0]
            if(ComNonAssNodes.size>0):
                comSizesTarget = int(np.floor(self.numNodes/self.numComs))* np.ones((self.numComs,)).astype(int) 
                comSizesTarget[-1] = self.numNodes-np.sum(comSizesTarget[:-1])
                preComSizes = np.sum(self.P, axis=1)
#                     TODO handle when comSizesTarget<preComSizes
                comSizeNeed = (comSizesTarget-preComSizes).astype(int)
                for i in np.arange(self.numComs):
                    fillIdx = np.random.choice(ComNonAssNodes, size=(comSizeNeed[i],), replace=False)
                    self.P[i, fillIdx] = 1 
                    ComNonAssNodes = list(set(ComNonAssNodes)-set(fillIdx))
#                     take care of the remaining nodes as a result of assigning some fine nodes to zero or multiple communities    
                ComNonAssNodes = np.argwhere(np.sum(self.P, axis=0)==0)[:,0]                        
                self.comIndices = [list(np.argwhere(np.squeeze(self.P[j, :])!=0).reshape(-1)) for j in np.arange(self.numComs)]
                self.comSizes = np.sum(self.P, axis=1).astype(int)
                if(np.any(np.array(aggregateList(self.comIndices))>self.numNodes) or np.sum(self.comSizes)!=self.numNodes):
                    raise ValueError('Invalid community to fine node mapping!')
        print('self.comSizes in SSBM core', self.comSizes)
        return  

def AdaptiveProfileChoose(population, size, maxColSum, df):
    if(True):
        prior_vec = np.array([1/(np.count_nonzero(vec)) for vec in population]) # 
        prior_vec = prior_vec/prior_vec.sum()
        prior_nu = np.array([1/nu for nu in (np.arange(df['nu'])+1)])
        prior_nu = prior_nu/prior_nu.sum()
    else:
        prior_vec = None
        prior_nu = None
    # Row-Wise selection
    choice_indices = np.random.choice(len(population), size, replace=True, p=prior_vec)
    normProfileMat = np.array([population[i] for i in choice_indices])
    if(maxColSum is not None):
        #TODO must complete
        colSums = np.sum(normProfileMat, 0)
        while((colSums>maxColSum).any()):
            print('normProfileMat Column Sums = {}, max Column Sum ={}'.format(colSums, maxColSum))
            if(True):
                #TODO can be done more efficiently
                colToReduce = np.argwhere(colSums>maxColSum)
                colToAdd = np.argwhere(colSums<maxColSum)
                for k in colToReduce:
                    whichNodeToReduce = np.argmax(normProfileMat[:, k]) #np.random.choice()
                    whichComToAdd = np.random.choice(np.argwhere(normProfileMat[whichNodeToReduce, colToAdd]==0)[:,0])
                    normProfileMat[whichNodeToReduce, whichComToAdd] = normProfileMat[whichNodeToReduce, k]
                    normProfileMat[whichNodeToReduce, k] = 0
                    
                
            else:
                normProfileMat = np.zeros_like(normProfileMat)
                # Column-Wise selection
                population_ColWise = [1/nu for nu in (np.array(df['nu'])+1)]
                inSize = size
                for k in np.arange(df['K']):
                    remaining_indices = np.argwhere(np)
                    choice_indices = np.random.choice(len(population_ColWise), size=inSize, replace=True, p=prior_nu)
                    normProfileMat[:,k] = population_ColWise[choice_indices]
                
                raise ValueError('Profile Matrix Invalid, implementation needed')
            colSums = np.sum(normProfileMat, 0)
            print('In Adaptive Profile Matrix Generation, column Sums = ', colSums)
            
        print('normProfileMat Column Sums = {}, max Column Sum ={}'.format(colSums, maxColSum))
    return normProfileMat
    
    
    
def profileMatGen(df, true_comIdx_fine=None):
    population = generateNormalizedProfileSet(df)
    if(true_comIdx_fine is None):
        maxColSum = int(df['n']/(df['K']))*np.ones((df['K'],))/df['r']
    else:
        trueComSizes = np.array([len(idx) for idx in true_comIdx_fine])
        print('In Profile MatGen: True community sizes = ', trueComSizes)
        maxColSum = trueComSizes/df['r']
        print('In Profile MatGen: coverage normalized true community sizes = maxColSum = ', maxColSum)
    normProfileMat = AdaptiveProfileChoose(population, size=df['m'], maxColSum=maxColSum, df=df)  
    return normProfileMat

class SBMLinCoarsMatCore():
    def __init__(self):
        return

def GenerateSBMLinCoarsMat(df, true_comIdx_fine=None, mesh=None):
    numFineNodes = df['n']
    numCoarseNodes = df['m']
    coverage = df['r']
    scaling = df['scaling']
    senseIdx = None
    B = None
    normProfileMat = None
    if(true_comIdx_fine is None):
        B = np.zeros((numCoarseNodes, numFineNodes))
        if(mesh is not None):
            chosen = choose2DRandomDistant(mesh, outSize=numCoarseNodes, coverageSize=coverage)
            for i in np.arange(numCoarseNodes):
                B[i, chosen[i]] = 1
            
            
    #         meshSensing = np.zeros_like(mesh)    
    #         meshSensing = 0    
    #         SBM_LinCoarsMat_core.meshSensing = meshSensing
        else:
            chosen = choose1DRandomDistant(np.arange(numFineNodes), size=numCoarseNodes, coverage=coverage)
            for i in np.arange(numCoarseNodes):
                B[i, chosen[i]:chosen[i]+coverage] = 1
    #         syncRatio = 1-count_non_sync(chosen, coverage, sizes_SBM)/total_num_communities
    else:
#         for k in np.arange(df['K']):
#             true_comIdx_fine[k] = np.array(true_comIdx_fine[k])
        normProfileMat = profileMatGen(df, true_comIdx_fine=true_comIdx_fine)
        # TODO can be done more efficiently
#         senseIdx = [list(itertools.chain.from_iterable([np.random.choice(list(comIdx[k]), size=int(normProfileMat[i,k]*df['r']**2), \
#                                                             replace=False) for k in np.arange(df['K'])])) for i in np.arange(df['m'])]      
        comIdx_fine_copy = true_comIdx_fine.copy()
        senseIdx = []
        for i in np.arange(df['m']):
            chosen = []
            for k in np.argwhere(normProfileMat[i,:]!=0).flatten():
                comIdx_fine_copy[k] = np.squeeze(np.array(list(comIdx_fine_copy[k])))
                if(not isinstance(comIdx_fine_copy[k], np.ndarray) or comIdx_fine_copy[k].size==1):
                    comIdx_fine_copy[k] = np.array([comIdx_fine_copy[k]])
#                 print('np.squeeze(np.array(comIdx_fine_copy[k])).shape = ', comIdx_fine_copy[k].shape)
                chosenSize = int(normProfileMat[i,k]*df['r'])
                if(comIdx_fine_copy[k].size>=chosenSize):
                    in_chosen = np.random.choice(comIdx_fine_copy[k], replace=False, size=chosenSize)
                else:
                    in_chosen = comIdx_fine_copy[k]
#                 (comIdx_fine_copy[k]).remove(chosen)
#                 print('comIdx_fine_copy[{}] = {}'.format(k, comIdx_fine_copy[k]))
                comIdx_fine_copy[k] = list(set(list(comIdx_fine_copy[k])).difference(set(in_chosen)))
                chosen.append(in_chosen)
            senseIdx.append(list(itertools.chain.from_iterable(chosen)))
            
    if(False):
        if(isinstance(scaling, str)):
            if(scaling == 'row_normalize'):
                row_sums = B.sum(axis=1)
                B = B / row_sums[:, np.newaxis]
            elif(mode.scaling == 'mat_normalize'):
                B = B / B.sum()
        else:
            B = B*scaling
    # len(aggregateList(senseIdx)) != len(set(aggregateList(senseIdx))) 
    return B, senseIdx, normProfileMat



       
class measuringCore():
    def __init__(self, df, mesh=None, true_comIdx_fine=None):
        self.linCoarseningMat, self.senseIdx, self.normProfileMat = GenerateSBMLinCoarsMat(df, true_comIdx_fine=true_comIdx_fine, mesh=mesh)
        return               
    
class CoarseningCommunityParams():
    def __init__(self, df, graphGenMode, measuring_core):
        self.graphGenMode = graphGenMode
        self.measuring_core = measuring_core
        return


def comProfileToNodeClustering(df, coms, coarseW):
    return NodeClustering(coms, graph=nx.convert_matrix.from_numpy_matrix(coarseW), \
                            method_name='groundTruth', overlap=True if df['nu']>1 else False)




def my_grid_search(graph, method, parameters, quality_score, aggregate=None):
    """
    Returns the optimal partition of the specified graph w.r.t. the selected algorithm and quality score.

    :param method: community discovery method (from nclib.community)
    :param graph: networkx/igraph object
    :param parameters: list of Parameter and BoolParameter objects
    :param quality_score: a fitness function to evaluate the obtained partition (from nclib.evaluation)
    :param aggregate: function to select the best fitness value. Possible values: min/max
    :return: at each call the generator yields a tuple composed by: the optimal configuration for the given algorithm, input paramters and fitness function; the obtained communities; the fitness score

    :Example:

    >>> import networkx as nx
    >>> from cdlib import algorithms, ensemble
    >>> g = nx.karate_club_graph()
    >>> resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
    >>> randomize = ensemble.BoolParameter(name="randomize")
    >>> communities, scoring = ensemble.grid_search(graph=g, method=algorithms.louvain,
    >>>                                                     parameters=[resolution, randomize],
    >>>                                                     quality_score=evaluation.erdos_renyi_modularity,
    >>>                                                     aggregate=max)
    >>> print(communities, scoring)
    """
    results = []
    counter = 1
    for communities in grid_execution(graph, method, parameters):
        in_dict = communities.method_parameters.copy()
        in_dict.update({"communities": communities, 'scoring': quality_score(graph, communities)})
        results.append(in_dict)
        print('grid search counter = ', counter)
        counter += 1
        
    argMax = np.argmax([res['scoring'].score for res in results])
    return results[argMax]
    



def evalCommunityRecovery(df, recoveredComIdx, trueComIdx, coarseW):
    """
    returns normalized F1 score, using the percentage of nodes in one partition vs the other
    """
    if(~isinstance(trueComIdx, NodeClustering)):
        trueComIdx = comProfileToNodeClustering(df, trueComIdx, coarseW)
    if(True):
        eval = dict(('Recovery Error with '+method, 1-evaluation.nf1(idx, trueComIdx).score) \
                                for method, idx in recoveredComIdx.items())
    else:
        eval = dict(('O-NMI Recovery Error with '+method, 1-evaluation.overlapping_normalized_mutual_information_MGH(idx, trueComIdx).score) \
                                for method, idx in recoveredComIdx.items()) 
    # evaluation.overlapping_normalized_mutual_information_LFK(recoveredComIdx, trueComIdx)
#     print('Community Evaluation = {}'.format(eval.score))
    print(eval)
    return eval
    
# import rpy2.robjects as robjects
# def wsbmCluster(df, W_tilde):
#     path = "WSSBM/"
#     robjects.r.source(path+"main.R")
#     L = np.max(W_tilde)
#     func = lambda x: x/L
#     p = robjects.r.wsbmCluster(W_tilde, df['K'], L, func) # robjects.r['wsbmCluster'](W_tilde, df['K'], L, Phi) input, output
#     return p
    
def recoverCommunities(df, W_tilde, true_comIdx_coarse=None):
    if(df['nu']==1):
        out_wsbmCluster = wsbmCluster(df, W_tilde)
        
#         ntrials = 100
#         L_ls = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
#         #L_ls = c(2, 3, 4, 5, 6, 8)
#         ham_mat = matrix(0, ntrials, length(L_ls))
#         
#         K = 3
#         n = 2100
#         
#         p = .3
#         q = .27
#         
#         Emat = matrix(q, n, n)
#         Dmat = diag(1, K) %x% matrix(p - q, n/K, n/K) 
#         Emat = Emat + Dmat
#         
#         for (it in 1:ntrials){
#             
#             tmp = rbinom(n=n*n, size=1, prob=matrix(Emat, n*n, 1))
#             A = matrix(as.numeric(tmp), n, n)
#             diag(A) = 0
#             Abin = A
#         
#             mask_mat = diag(1, K) %x% matrix(1, n/K, n/K)
#         
#             nne1 = sum(sum(A[mask_mat == 1 & A == 1] ))
#             nne2 = sum(sum(A)) - nne1
#         
#             A[mask_mat == 1 & A == 1] = rnorm(nne1, 0.3, .8)
#             A[mask_mat == 0 & A == 1] = rnorm(nne2, 0, 1)
#         
#             true_cluster = matrix(1:K, K, 1) %x% matrix(1, n/K, 1)
#         
#             for (il in 1:length(L_ls)){
#                
#                 L = L_ls[il]
#                 Phi = function(x) { plaplace(x, m=0, s=2) }
#                 res = wsbmCluster(A, K, L, Phi)
#                 ham = hammingDist(true_cluster, res$clustering, K)/n
#                 ham_mat[it, il] = ham
#         
#                 ham2 = hammingDist(true_cluster, res$spectral, K)/n
#                 
#                 print(paste(it, L, round(ham, 4), round(res$discrep, 4), round(ham2, 4)))   
#             }
#             save(ham_mat, file="tmp.RData")
#         }       
        comIdx = 0
        clustering_obj = 0
    else:
        __all__ = ["ego_networks", "demon", "angel", "node_perception", "overlapping_seed_set_expansion", "kclique", "lfm",
           "lais2", "congo", "conga", "lemon", "slpa", "multicom", "big_clam", "danmf", "egonet_splitter", "nnsed",
           "nmnf", "aslpaw", "percomvc"]
#         algorithms.conga(g, number_communities=df['K'])
#         algorithms.slpa(g)
#         algorithms.nnsed(g) # --> best thus far
#         algorithms.slpa(g) # , T=100, r=0.3 # 
#         algorithms.overlapping_seed_set_expansion(g, seeds=np.arange(W_tilde_b.shape[0]).tolist())
#         algorithms.lemon(g, seeds=np.arange(W_tilde_b.shape[0]))
#         algorithms.lfm(g, alpha=1)
#         algorithms.lais2(g)
#         algorithms.big_clam(g)
#         algorithms.node_perception(g, threshold=0.25, overlap_threshold=0.25)
#         algorithms.angel(g, threshold=0.5, min_community_size=df['K']) 
#         algorithms.ego_networks(g) 
#         algorithms.demon(g, epsilon =0.25) #, min_com_size=df['K'] 
        threshold = (df['r']**2)*(df['tauTilde']*df['p']+(1-df['tauTilde'])*df['q'])
        mean_edge_val = np.mean(W_tilde)
        min_edge_val = np.min(W_tilde)
        if(threshold>mean_edge_val):
            th_step = (threshold-mean_edge_val)/10
        else:
            th_step = (threshold-min_edge_val)/10
        grid_search_flag = True
        grid_optimal_found = True
        counter = 1
        while(True):
            print('threshold={}, mean edge value={}, min edge value={}'.format(threshold, mean_edge_val, min_edge_val))
            W_tilde_b = (W_tilde>=threshold).astype(int)
            g = nx.convert_matrix.from_numpy_matrix(W_tilde_b)
            if(true_comIdx_coarse is not None):
                g.communities = true_comIdx_coarse
            if(False):
                nx.draw(g)
                plt.show()
            try:
                print('Connected Components = ', list(nx.connected_components(g)))
                print('Before community recovery ...')
                min_comm_size = df['K'] # df['min Community Size']
                if(grid_search_flag):
                    dimensions = ensemble.Parameter(name="dimensions", start=df['K']*2, end=int(df['m']/5), step =int(((df['m']/5)-df['K']*2)/3))
                    iterations = ensemble.Parameter(name="iterations", start=int(500.0), end=int(500.0)) # ensemble.Parameter(name="iterations", start=int(20.0), end=int(500.0), step =int(200.0))
                else:
                    dimensions = int(df['m']/10) if df['m']>=100 else np.max([df['K']*2, int(df['m']/10)])
                    iterations = 500
                if(grid_search_flag):
                    if(not grid_optimal_found):
                        lambd = ensemble.Parameter(name="lambd", start=0, end=0.4, step =0.2)
                        alpha = ensemble.Parameter(name="alpha", start=0, end=0.2, step =0.05)
                        beta = ensemble.Parameter(name="beta", start=0, end=0.08, step =0.005)
                        eta = ensemble.Parameter(name="eta", start=0.1, end=10.0, step =1.0)
                    else:
                        lambd = ensemble.Parameter(name="lambd", start=0.2, end=0.2)
                        alpha = ensemble.Parameter(name="alpha", start=0.1, end=0.1)
                        beta = ensemble.Parameter(name="beta", start=0.02, end=0.02)
                        eta = ensemble.Parameter(name="eta", start=0.1, end=0.1)
                    clusters = ensemble.Parameter(name="clusters", start=df['K'], end=df['K'])
                    dict_results = my_grid_search(g, method=algorithms.nmnf, parameters=[clusters,dimensions,lambd,alpha,beta,iterations,eta], \
                                                                            quality_score=evaluation.nf1, aggregate=max)
                    coms = dict_results['communities']
                    print('M-NMF optimal parameters = ', dict_results)
                else:
                    # beta =0.005 works better than 0.05, eta=1.0 works better than 5.0
                    coms = algorithms.nmnf(g, clusters=df['K'], dimensions=dimensions, lambd=0.2, alpha=0.1, beta=0.005, \
                                                                        iterations=iterations, eta=1.0)
                    
                clustering_obj = {'M-NMF': coms} 
                print('M-NMF Done.')
                try:
                    if(grid_search_flag):
                        if(not grid_optimal_found):
                            r = ensemble.Parameter(name="r", start=0.01, end=0.6, step =0.1)
                        else:
                            r = ensemble.Parameter(name="r", start=0.1, end=0.1)
                        t = ensemble.Parameter(name="t", start=int(500.0), end=int(500.0))
                        dict_results = my_grid_search(g, method=algorithms.slpa, parameters=[t, r], \
                                                         quality_score=evaluation.nf1, aggregate=max)
                        coms = dict_results['communities']
                        print('SLPA optimal parameters = ', dict_results)
                    else:
                        # r=0.1 works better than 0.8
                        coms = algorithms.slpa(g, t=iterations, r=0.1)
                        
                    clustering_obj['SLPA'] = coms
                    print('SLPA Done.')
                except Exception as exc:
                    print(exc)
                    pass
#                 try:
#                     clustering_obj['Conga'] = algorithms.conga(g, number_communities=df['K'])
#                     print('Conga Done.')
#                 except:
#                     pass
                try:
                    if(grid_search_flag):
                        dict_results = my_grid_search(g, method=algorithms.nnsed, parameters=[dimensions, iterations], \
                                                                quality_score=evaluation.nf1, aggregate=max)
                        coms = dict_results['communities']
                        print('nnsed optimal parameters = ', dict_results)
                    else:
                        coms = algorithms.nnsed(g, dimensions=dimensions, iterations=iterations) # dimensions=32, iterations=10
                        
                    clustering_obj['nnsed'] = coms
                    print('nnsed Done.')
                except:
                    pass 
                try:
                    if(grid_search_flag):
                        if(not grid_optimal_found):
                            learning_rate = ensemble.Parameter(name="learning_rate", start=5e-5, end=5e-4, step =5e-4)
                        else:
                            learning_rate = ensemble.Parameter(name="learning_rate", start=5e-5, end=5e-5) 
                        dict_results = my_grid_search(g, method=algorithms.big_clam, parameters=[dimensions, iterations, learning_rate], \
                                                                quality_score=evaluation.nf1, aggregate=max)
                        coms = dict_results['communities']
                        print('BigClam optimal parameters = ', dict_results)
                    else:
                        # learning_rate = 0.0001 works better than 0.005
                        coms = algorithms.big_clam(g, dimensions=dimensions, iterations=iterations, learning_rate=0.0001) # dimensions=8, iterations=50
                    
                    clustering_obj['BigClam'] = coms
                    print('BigClam Done.')
                except:
                    pass
#                 try:
#                     clustering_obj['Node Perception'] = algorithms.node_perception(g, threshold=0.25, overlap_threshold=0.25, min_comm_size=min_comm_size)
#                 except:
#                     pass
#                 try:
#                     clustering_obj['ego networks'] = algorithms.ego_networks(g) 
#                 except:
#                     pass
#                 try:
#                     clustering_obj['Overlapping Seed Set Expansion'] = algorithms.overlapping_seed_set_expansion(g, seeds=np.arange(W_tilde_b.shape[0]).tolist())
#                 except:
#                     pass
#                 try:
#                     clustering_obj['lemon'] = algorithms.lemon(g, seeds=np.arange(W_tilde_b.shape[0]))
#                 except:
#                     pass
#                 try:
#                     clustering_obj['LFM'] = algorithms.lfm(g, alpha=1) 
#                 except:
#                     pass
#                 try:
#                     clustering_obj['Lais2'] = algorithms.lais2(g)
#                 except:
#                     pass
#                 try:
#                     clustering_obj['angel'] = algorithms.angel(g, threshold=0.5, min_community_size=min_comm_size) 
#                 except:
#                     pass
#                 try:
#                     clustering_obj['demon'] = algorithms.demon(g, epsilon =0.25)
#                 except:
#                     pass
                print('After community recovery  ...')
                return clustering_obj
            except Exception as exc:
#                 print(traceback.format_exc())
                print(exc)
                print('Connected Components = ', nx.connected_components(g))
                raise ValueError('Error While Recovering Communities!')
                if(False):
                    print('Connected Components = ', nx.connected_components(g))
                    counter += 1
                    threshold -= th_step
                    print('Graph is not connected, trying # {} for another threshold  ...'.format(counter))
                    continue
                if(False):
                    print('Graph is not connected, communities are recovered from components ...')
                    con_comp = nx.connected_components(g)
                    disjointed_nodes = list(con_comp)
                    print('disjointed_nodes = ', disjointed_nodes)
                    len_comp = [len(c) for c in sorted(con_comp, key=len, reverse=True)]
                    print('component lengths = ', len_comp)
                    S = [g.subgraph(c).copy() for c in nx.connected_components(g) if len(c)>2]
                    comIdx = []
                    for dis_g in S:
        #                 print('W_tilde_b[dis_nodes,dis_nodes] = ', W_tilde_b[dis_nodes[:, None],dis_nodes])
        #                 dis_g = nx.convert_matrix.from_numpy_matrix(W_tilde_b[dis_nodes[:, None],dis_nodes])
                        print('dis_g =', dis_g)
                        print('dis_g size =', dis_g.size())
                        in_obj = algorithms.nmnf(dis_g, clusters=df['K']) 
                        print('in_obj = ', in_obj)
                        print('in_obj.communities = ', in_obj.communities)
                        comIdx.extend(in_obj.communities)
                        print('comIdx = ', comIdx)
#         comIdx = coms.communities # .to_node_community_map()
    return clustering_obj


def mat2List(mat):
    return [list(np.argwhere(mat[:,k]).flatten()) for k in np.arange(mat.shape[1])]    


def min_coverageSBM(df): # condition on r to satisfy x^3- ax^2 -b>0
    fracN = np.log(df['n'])/df['n']
    rho_1 = (df['alpha']-df['beta'])*(1/df['nu']-df['tauTilde'])/(((df['alpha']-df['beta'])/df['nu']+df['beta'])**(1/2))
    rho_2 =  (df['alpha']/df['beta']-1)*df['tauTilde']
    rho_3 = 0.7915*(
        1/((np.min(
            [df['alpha']*(1-df['alpha']*fracN)/df['nu'] + df['beta']*(1-df['beta']*fracN)*(1-1/df['nu']),df['alpha']*(1-df['alpha']*fracN)]
            )  )**(1/2))
        + 1/(df['beta']*(1-df['beta']*fracN))
        )
    a = rho_3
    b = 2*rho_3/((np.max([rho_1,rho_2]))**2)
    sharedTermIn = (3* (3**(1/2)) * ((4*(a**3)*(b)+27*(b**2))**(1/2)) + 2*(a**3) + 27*b)**(1/3)
    minC = ((sharedTermIn)/((2)**(1/3))+((2)**(1/3)*(a**2))/(sharedTermIn)+a)/(3*fracN)
    if(isnan(minC)):
         ValueError('Invalid Values to Compute Min Coverage Size')
    return int(np.ceil(minC))
