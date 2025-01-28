import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from lib import utils
from lib.DCF import *


def logpdf_GAU_ND_1sample(x, mu, C):
    M=x.shape[0]
    logdet=np.linalg.slogdet(C)[1]
    cInv= np.linalg.inv(C)
    firstTerm= -M/2*np.log(2*np.pi)
    
    xc= x-mu
    thirdTerm= np.dot(xc.T, np.dot(cInv,xc)).ravel()
    return firstTerm - 0.5*logdet - 0.5*thirdTerm
    
def logpdf_GAU_ND_slow(X, mu, C):
    
    Y=[]
    
    for i in range(X.shape[1]):
        
        Y.append(logpdf_GAU_ND_1sample(X[:, i:i+1], mu, C))
        
    return np.array(Y).ravel()

def logpdf_GAU_ND(x, mu, C):

    P = np.linalg.inv(C)

    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

def loglikelihood(X, m_ML, C_ML):
    
    return logpdf_GAU_ND(X, m_ML, C_ML).sum()

def plot_density(D, label):

    classdic = {0: "Counterfeit",
                1: "Genuine"}

    for i in range(D.shape[0]):
        Dfeat = D[i:i+1,:]
        mu, C = utils.compute_mu_C(Dfeat)
    

        plt.figure()
        plt.hist(Dfeat.ravel(), bins=50, density=True)
        XPlot = np.linspace(-8, 12, 1000)
        plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(utils.vrow(XPlot), mu, C)))
        plt.title(f'Estimated density for class {classdic[label]} and feature {i}')
        plt.grid(True)
        plt.savefig(f'project\plots\MVG\ML_{classdic[label]}_{i}.pdf')
        

def compute_per_feature_densities(D, label):

    densities = []

    for i in  range(D.shape[0]):    

        mu, C = utils.compute_mu_C(D[i:i+1,:])
        densities.append(logpdf_GAU_ND(D[i:i+1,:], mu, C))


    plot_density(D, label)

    return densities


def compute_per_class_densities(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    densities0 = compute_per_feature_densities(D0, 0)
    densities1 = compute_per_feature_densities(D1, 1)

    return densities0, densities1

def ml_estimates_gau(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    Dlist = [ D0, D1]
    mulist = []
    Clist = []

    for i in range(len(Dlist)):

        mu, C = utils.compute_mu_C(Dlist[i])
        mulist.append(mu)
        Clist.append(C)

    return mulist, Clist

def compute_llr(DTE, mulist, Clist):
    
    S = np.zeros((2,DTE.shape[1]))
    
    for i in range(2):
        for j in range(DTE.shape[1]):
            S[i][j] = np.exp(logpdf_GAU_ND_1sample(DTE[:,j:j+1], mulist[i], Clist[i]))
    
    llr = np.log(S[1]/S[0])

    return llr

def compute_ER(pred_result):

    return pred_result.count(False)/len(pred_result)

def MVG_classifier(DTR, LTR, DTE, LTE, true_prior):
    
    mulist, Clist = ml_estimates_gau(DTR, LTR)

    llr = compute_llr(DTE, mulist, Clist)
    false_prior = 1 - true_prior
    t = -np.log(true_prior/false_prior)

    pred_result = utils.prediction_test(LTE, llr, t)

    error_rate = compute_ER(pred_result)

    print(f"Error rate of MVG classifier: {error_rate}")

    return mulist, Clist

def compute_CTied(C_v, DTR, LTR):
    
    C_tied=np.zeros((C_v[0].shape[0],C_v[0].shape[1]))
    for i in range(len(C_v)):
        
        DTRi=DTR[:, LTR==i]
        C_tied=C_tied+(C_v[i]*DTRi.shape[1])

    CTied= C_tied/DTR.shape[1]
    CTiedl = []

    for i in range(2):
        CTiedl.append(CTied)

    return CTiedl

def MVG_Tied_classifier(DTR, LTR, DTE, LTE, true_prior):
    
    mulist, Clist = ml_estimates_gau(DTR, LTR)

    CTiedl = compute_CTied(Clist, DTR, LTR)
        

    llr = compute_llr(DTE, mulist, CTiedl)
    false_prior = 1 - true_prior
    t = -np.log(true_prior/false_prior)

    pred_result = utils.prediction_test(LTE, llr, t)

    error_rate = compute_ER(pred_result)

    print(f"Error rate of MVG Tied classifier: {error_rate}")

    return mulist, Clist

def compute_CNB(Clist):

    CNaive = []
    for C in Clist:
        CNaive.append(utils.diagonalize(C))

    return CNaive

def Naive_Bayes_classifier(DTR, LTR, DTE, LTE, true_prior):
    
    mulist, Clist = ml_estimates_gau(DTR, LTR)

    CNaivel = compute_CNB(Clist)
    
    llr = compute_llr(DTE, mulist, CNaivel)
    false_prior = 1 - true_prior
    t = -np.log(true_prior/false_prior)

    pred_result = utils.prediction_test(LTE, llr, t)

    error_rate = compute_ER(pred_result)

    print(f"Error rate of Naive Bayes classifier: {error_rate}")

    return mulist, Clist

def MVG_DCF(DTR, LTR, DTE, LTE, working_point, model="MVG", verbose=False, plot=False):
    
    mulist, Clist = ml_estimates_gau(DTR, LTR)

    if model == "Tied":
        Clist = compute_CTied(Clist, DTR, LTR)
    elif model == "Naive":
        Clist = compute_CNB(Clist)

    llr = compute_llr(DTE, mulist, Clist)
    #eff_prior = DCF.compute_effective_prior(working_point)

    M, DCF, minDCF = compute_DCF_minDCF(llr, LTE, working_point)
    if verbose:
        print(f" Results for {model} are: M: {M}, DCF: {DCF}, and minDCF: {minDCF}")
    if plot:
        Bayes_error_plots(llr, LTE, model)
        
    return DCF, minDCF


