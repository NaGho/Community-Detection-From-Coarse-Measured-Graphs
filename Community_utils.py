import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import sys
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
import rpy2.robjects as robjects
import networkx as nx
from cdlib import algorithms, ensemble, evaluation, classes
from cdlib.classes import NodeClustering

# pd.set_option('display.width', 1000)
def nCr(n,r):
    f = factorial
    return f(n) / f(r) / f(n-r)


def profile2community(phi, nu):
    return

def CH_divergence(c1,c2):
    t = 1/2
    return np.sum(t*c1+(1-t)*c2-(c1**t)*(c2**(1-t)))
    
def generateNormalizedProfileSet(df):
#     phiN = np.zeros((df['K_nu'], df['K']))
#     for nu in np.arange(1, df['nu']):
#         for k in np.arange():
#             phiN[,]
    lst = list(itertools.product([0, 1], repeat=df['K']))
    phiN = np.array([s for s in lst if (np.sum(s)>0 and np.sum(s)<=df['nu'])])
    row_sums = phiN.sum(axis=1)
    phiN = phiN / row_sums[:, np.newaxis]
    return phiN
    
    
    
def UB_ML_Failure_Error_Function(df):
    if(df['nu']==1): # (False): # 
        g = int(df['L']/df['K'])
#         err_terms1 = [-np.inf]
#         err_terms2 = [-np.inf]
        I = -2*(df['C']**2)*np.log(np.sqrt((1-df['alpha']*np.log(df['n'])/df['n'])*(1-df['beta']*np.log(df['n'])/df['n'])) + \
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
        logErr = np.log(min_err)
        if(logErr>0):
            logErr = 0
    else:
        tau_tilde = 1e-2 # 1/(df['nu']+1)
        df['K_nu'] = 0
        for nu in np.arange(df['nu'])+1:
            df['K_nu'] += int(nCr(df['K'], nu))
        phiN = generateNormalizedProfileSet(df)
        U = np.zeros((df['K_nu'],df['K_nu']))
        epsilonMax = np.zeros((df['K_nu'],df['K_nu']))
        logDivn = np.log(df['n'])/df['n']
        for k in np.arange(df['K_nu']):
            for kprime in np.arange(k, df['K_nu']):
                normDot = np.dot(phiN[k,:],phiN[kprime,:])
                U[k,kprime] = norm.cdf(
                                (df['alpha']-df['beta'])
                                *(tau_tilde-normDot*df['C']*logDivn)
                                /((df['alpha']-df['beta'])*normDot + df['beta'])
                                )
                epsilonMax[k,kprime] = 0.7915/(
                    (df['alpha']*(1-df['alpha']*logDivn)*normDot+df['beta']*(1-df['beta']*logDivn)*(1-normDot))
                    *logDivn*df['C']**2
                    )     
        logErrArray = []
        for epCoeff in [-1, 0, 1]:
            logErr = 0 
            for k in np.arange(df['K_nu']):
                for kprime in np.arange(k+1, df['K_nu']):
                    logErr += df['L']**(-CH_divergence(df['L']*(U[k,:]+epCoeff*epsilonMax[k,:])
                                               /(np.log(df['L'])*df['K_nu']),df['L']*(U[kprime,:]+epCoeff*epsilonMax[kprime,:])
                                                       /(np.log(df['L'])*df['K_nu'])))
            logErr = np.log(logErr)
            if(logErr>0):
                logErr = 0
#         logErr = np.sum([[L**(-CH_divergence(df['L']*U[k,:]/np.log(df['L']),df['L'])*U[kprime,:]/np.log(df['L']))\
#                           for k in np.arange(df['K_nu'])] for kprime in np.arange(df['K_nu'])])
            logErrArray.append(logErr)
            df['ML Failure UB Log-Error (Lowest U)'] = logErrArray[0]
            df['ML Failure UB Log-Error (Highest U)'] = logErrArray[-1]
            logErr = logErrArray[1]
    df['ML Failure UB Log-Error'] = logErr
    return logErr, df


class SSBMCore():
    def __init__(self, df, measuring_core, mesh):
        self.numNodes = df['n']
        self.numComs = df['K']
        self.Qprobs = (df['alpha']*np.ones((self.numComs,self.numComs)) + (df['beta']-df['alpha']) 
                                    * np.diag(np.ones(self.numComs,)))*np.log(df['n'])/df['n']
        self.P = np.zeros((self.numComs, self.numNodes))
        
        self.normProfileMat = profileMatGen(df)
        
        if(measuring_core is None or measuring_core.linCoarseningMat is None):
            self.comSizes = int(np.floor(self.numNodes/self.numComs)) * np.ones((self.numComs,)).astype(int) # [50, 50, 100, 100, 150]
            self.comSizes[-1] = n-np.sum(self.comSizes[:-1])
            self.comIndices = []
            remaining_indices = np.arange(mesh.size)
            for i in np.arange(self.numComs):
                in_Chosen = np.random.choice(remaining_indices, size=self.comSizes[i], replace=False)
                self.P[i,in_Chosen] = 1
                remaining_indices = list(set(remaining_indices)-set(in_Chosen))
                in_Chosen.sort()
                self.comIndices.append(in_Chosen)
                
        else:
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
                
#             else:
#                 sizes_SBM_indices = np.concatenate(([0], np.cumsum(self.comSizes)))
#                 self.comIndices = [np.arange(sizes_SBM_indices[i],sizes_SBM_indices[i+1]) for i in np.arange(self.numComs)]
        return  
    
def profileMatGen(df):
    population = generateNormalizedProfileSet(df)
    prior_vec = np.array([1/(np.count_nonzero(vec)) for vec in population])
    prior_vec = prior_vec/prior_vec.sum()
    choice_indices = np.random.choice(len(population), df['L'], replace=True, p=prior_vec) # , p=list_of_prob
    choices = [population[i] for i in choice_indices]
    return np.array(choices)
                 
class measuringCore():
    def __init__(self, df, linCoarseningMat):
        self.linCoarseningMat = linCoarseningMat
        return               
    
class CoarseningCommunityParams():
    def __init__(self, df, graphGenMode, measuring_core):
        
        self.graphGenMode = graphGenMode
        self.measuring_core = measuring_core
        return


def comProfileToNodeClustering(df, coms):
    return NodeClustering(coms, graph=nx.convert_matrix.from_numpy_matrix(np.zeros((df['L'],df['L']))), \
                                                              method_name='groundTruth', overlap=True if df['nu']>1 else False)



def evalCommunityRecovery(df, recoveredComIdx, trueComIdx):
    """
    returns normalized F1 score, using the percentage of nodes in one partition vs the other
    """
    if(~isinstance(trueComIdx, NodeClustering)):
        trueComIdx = comProfileToNodeClustering(df, trueComIdx)
    eval = evaluation.nf1(recoveredComIdx, trueComIdx) # evaluation.overlapping_normalized_mutual_information_LFK(recoveredComIdx, trueComIdx)
    print('Community Evaluation = {}'.format(eval.score))
    return eval.score
    

def wsbmCluster(df, W_tilde):
    path = "WSSBM/"
    robjects.r.source(path+"main.R")
    L = np.max(W_tilde)
    func = lambda x: x/L
    p = robjects.r.wsbmCluster(W_tilde, df['K'], L, func) # robjects.r['wsbmCluster'](W_tilde, df['K'], L, Phi) input, output
    return p
    
def recoverCommunities(df, W_tilde):
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
    else:
        W_tilde_b = (W_tilde>(df['C']**2)*(df['tauTilde']*df['p']+(1-df['tauTilde'])*df['q'])).astype(int)
        g = nx.convert_matrix.from_numpy_matrix(W_tilde_b)
        clustering_obj = algorithms.demon(g, epsilon =0.25) #, min_com_size=df['K'] # algorithms.slpa(g) # , T=100, r=0.3 # 
#         comIdx = coms.communities # .to_node_community_map()
    return clustering_obj
    