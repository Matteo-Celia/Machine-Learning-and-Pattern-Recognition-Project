import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.optimize
import itertools
from lib.DCF import *
from lib.utils import * 
from lib.Gaussian_models import logpdf_GAU_ND

def logpdf_GMM(X, gmm):

    S = np.zeros((len(gmm), X.shape[1]))
    for idx,params in enumerate(gmm):
        S[idx, :] = logpdf_GAU_ND(X, params[1] , params[2]) + np.log(params[0])

    return scipy.special.logsumexp(S, axis=0)

def compute_gmm_scores(D, gmm):
    
    llr = logpdf_GMM(D, gmm[1]) - logpdf_GMM(D, gmm[0])
    return llr


def GMM_tied_transformation(gmm):
    tied_sigma = np.zeros((gmm[0][2].shape[0], gmm[0][2].shape[0]))
    for g in range((len(gmm))):
        tied_sigma += gmm[g][2] * gmm[g][0] #z_vec[g]
    #tied_sigma = (1 / n) * tied_sigma
    for g in range((len(gmm))):
        gmm[g] = (gmm[g][0], gmm[g][1], tied_sigma)
    return gmm

def GMM_diag_tranformation(gmm):
    for g in range((len(gmm))):
        Sigma_g = gmm[g][2] * np.eye(gmm[g][2].shape[0])
        gmm[g] = (gmm[g][0], gmm[g][1], Sigma_g)
    return gmm

def LBG_algorithm(iterations, X, init_gmm, alpha, psi, variant=None):

    if variant == "Diag":
        init_gmm = GMM_diag_tranformation(init_gmm)
    elif variant == "Tied":
        init_gmm = GMM_tied_transformation(init_gmm)

    for i in range(len(init_gmm)):
        covNew = init_gmm[i][2]
        U, s, _ = np.linalg.svd(covNew)
        s[s < psi] = psi
        covNew = np.dot(U, vcol(s) * U.T)
        init_gmm[i] = (init_gmm[i][0], init_gmm[i][1], covNew)

    init_gmm = em_algorithm(X, init_gmm, psi, variant)

    for i in range(iterations):
        gmm_new = list()
        for g in init_gmm:
            sigma_g = g[2]
            U, s, _ = np.linalg.svd(sigma_g)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            new_w = g[0]/2
            gmm_new.append((new_w, g[1] - d, sigma_g))
            gmm_new.append((new_w, g[1] + d, sigma_g))
        init_gmm = em_algorithm(X, gmm_new, psi, variant)
    return init_gmm

def em_algorithm(X, gmm, psi, variant=None):
    ll_new = None
    ll_old = None
    while ll_old is None or ll_new - ll_old > 1e-6:
        ll_old = ll_new
        s_joint = np.zeros((len(gmm), X.shape[1]))
        for g in range(len(gmm)):
            s_joint[g, :] = logpdf_GAU_ND(
                X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
        s_marginal = scipy.special.logsumexp(s_joint, axis=0)
        ll_new = s_marginal.sum() / X.shape[1] #average log likelihood
        P = np.exp(s_joint - s_marginal)
        gmm_new = []
        z_vec = np.zeros(len(gmm))
        for g in range(len(gmm)):
            gamma = P[g, :]
            zero_order = gamma.sum()
            z_vec[g] = zero_order
            first_order = (vrow(gamma) * X).sum(1)
            second_order = np.dot(X, (vrow(gamma) * X).T)
            w = zero_order / X.shape[1]
            mu = vcol(first_order / zero_order)
            sigma = second_order / zero_order - np.dot(mu, mu.T)
            gmm_new.append((w, mu, sigma))

        if variant == "Diag":
            gmm_new = GMM_diag_tranformation(gmm_new)
        elif variant == "Tied":
            gmm_new = GMM_tied_transformation(gmm_new)

        for i in range(len(gmm)):
            covNew = gmm_new[i][2]
            U, s, _ = np.linalg.svd(covNew)
            s[s < psi] = psi
            covNew = np.dot(U, vcol(s) * U.T)
            gmm_new[i] = (gmm_new[i][0], gmm_new[i][1], covNew)
        gmm = gmm_new
    return gmm


def GMM_DCF(working_point, llr, L):

    M = Bayes_decision_binary(working_point, llr, L)
    
    DCFu = Bayes_risk(M, working_point)

    DCF = normalize_DCF(DCFu,working_point)
    
    minDCF = compute_minDCF(llr, L, working_point)
    #print(f"Results for actual DCF={DCF} and minDCF={minDCF}")
    return DCF,minDCF

def GMM(DTR,LTR,DTE,LTE,piT,variant,components):
    folder_path = 'project/saved_models/GMM/'
    wp = [piT,1,1]
    D0 = DTR[:,LTR==0]
    D1 = DTR[:,LTR==1]
    mu0,C0 = compute_mu_C(D0)
    mu1,C1 = compute_mu_C(D1)
    gmm0 =[(1.0, mu0, C0)]
    gmm1 =[(1.0, mu1, C1)]
    iterations = int(np.log2(components))

    gmm_new0 = LBG_algorithm(iterations,D0,gmm0,psi=0.01,alpha=0.1,variant=variant)
    gmm_new1 = LBG_algorithm(iterations,D1,gmm1,psi=0.01,alpha=0.1,variant=variant)
    gmm_l = [gmm_new0, gmm_new1]

    llr = compute_gmm_scores(DTE,gmm_l)
    DCF, minDCF = GMM_DCF(wp,llr,LTE)
    
    print(f"Results for GMM {variant} with {components} components and piT={piT} actual DCF={DCF} and minDCF={minDCF}")

    path = folder_path +'GMM_Scores_' + str(variant)+ '_piT=' + str(piT) + '_components='+str(components)
    np.save(path, llr)
    path = folder_path +'GMM_parameters_gmm0_' + str(variant)+ '_piT=' + str(piT) + '_components='+str(components)+".json"
    save_gmm(gmm_new0,path)
    path = folder_path +'GMM_parameters_gmm1_' + str(variant)+ '_piT=' + str(piT) + '_components='+str(components)+".json"
    save_gmm(gmm_new1,path)
    return DCF, minDCF

def train_combinations(DTE,LTE,components_list,variant,piT):
    folder_path = 'project/saved_models/GMM/'
    wp = [piT,1,1]
    for combination in itertools.permutations(components_list, 2):
        comp_c0, comp_c1 = combination
        params_GMM0 = load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm0_{str(variant)}_piT=0.1_components={str(comp_c0)}.json")
        params_GMM1 = load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm1_{str(variant)}_piT=0.1_components={str(comp_c1)}.json")
        gmm_l = [params_GMM0,params_GMM1]
        llr = compute_gmm_scores(DTE,gmm_l)
        folder_path = 'project/saved_models/GMM/'
        path = folder_path +'GMM_Scores_' + str(variant)+ '_piT=' + str(piT) + '_componentsC0='+str(comp_c0)+ '_componentsC1='+str(comp_c1)
        np.save(path, llr)
        DCF, minDCF = GMM_DCF(wp,llr,LTE)
        print(f"Results for GMM {variant} with {comp_c0} components for class 0 and {comp_c1} components for class 1 and piT={piT} actual DCF={DCF} and minDCF={minDCF}")
        

def plot_DCF_GMM(componenents_list, DCFlist, minDCFlist, variant, piT):

    
    plt.figure()
    plt.plot(componenents_list, DCFlist, label='actual DCF',color='r')
    plt.plot(componenents_list, minDCFlist, label='min DCF',color='b')
    #plt.xscale('log', base=10)
    plt.xticks(componenents_list)
    plt.legend(['DCF','min DCF'])
    plt.xlabel('# components')
    
    plt.title(f'actual and min DCF as function of # components in GMM:')
    
    plt.savefig(f'project\plots\GMM\DCF_minDCF_{variant}_piT={piT}.pdf')
            
