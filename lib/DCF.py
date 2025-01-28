import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from lib.utils import *

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(np.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return np.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return np.argmin(expectedCosts, 0)

# Build uniform cost matrix with cost 1 for all kinds of error, and cost 0 for correct assignments
def uniform_cost_matrix(nClasses):
    return np.ones((nClasses, nClasses)) - np.eye(nClasses)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return np.int32(llr > th)

# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / np.min(costMatrix @ vcol(prior_array))
    return bayesError

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)
        
    
    
# Compute minDCF (slow version, loop over all thresholds recomputing the costs)
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = np.concatenate([np.array([-np.inf]), llrSorted, np.array([np.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = np.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut) # we return also the corresponding thresholds
    
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one


def BEP_cal(logOdds, actDCF_precal,actDCF_cal, minDCF,eval, model=""):
    
    
    plt.figure()
    if model == "fusion_all":
        plt.plot(logOdds, actDCF_cal, label='actDCF', color='r')
        plt.plot(logOdds, minDCF, linestyle='--',label='min DCF', color='b')
    else:
        if actDCF_precal is not None:
            plt.plot(logOdds, actDCF_precal,  linestyle=':',label='actDCF (pre-cal.)', color='r')
        plt.plot(logOdds, actDCF_cal, label='actDCF (cal.)', color='r')
        if eval:
            plt.plot(logOdds, minDCF, linestyle='--',label='min DCF', color='b')
        else:
            plt.plot(logOdds, minDCF, linestyle='--',label='min DCF (pre-cal.)', color='b')

    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    #plt.legend(['DCF','min DCF'])
    plt.xlabel('Effective prior log odds')
    plt.legend()
    plt.title(f'Bayes error plot for {model} :')
    if eval:
        plt.savefig(f'project\plots\BayesErrorPlots\BEP_cal_eval_{model}.pdf')
    else:
        plt.savefig(f'project\plots\BayesErrorPlots\BEP_cal_{model}.pdf')
    
        
    #return actDCF, minDCF


#//////////////////////

def compute_effective_prior(working_point):

    true_prior, Cfn, Cfp = working_point

    num = true_prior*Cfn
    den = true_prior*Cfn + ((1-true_prior)*Cfp)
    eff_prior = num / den
    return eff_prior

def Bayes_decision_binary(working_point, llr, L, threshold=None):
    
    piT, Cfn, Cfp = working_point
    
    if threshold:
        t=threshold
    else:
        t=-np.log((piT*Cfn)/((1-piT)*Cfp))
    
    predicted=np.zeros((np.size(llr)))
    
    conf_matrix = np.zeros((2,2))
    
    for i in range(np.size(llr)):
        predicted[i]= 1 if llr[i]> t else 0
    
    for i in range(np.size(L)):
        
        conf_matrix[int(predicted[i]),L[i]]+=1
       
    return conf_matrix

def Bayes_risk(M, working_point):
    
    piT, Cfn, Cfp =working_point[0], working_point[1], working_point[2]
    
    FNR=M[0,1]/(M[0,1]+M[1,1])
    FPR=M[1,0]/(M[0,0]+M[1,0])
    
    DCFu=piT*Cfn*FNR+(1-piT)*Cfp*FPR
    
    return DCFu

def normalize_DCF(DCFu, working_point):
    
    piT, Cfn, Cfp =working_point[0], working_point[1], working_point[2]
    Bdummy= np.minimum(piT*Cfn,(1-piT)*Cfp)
    
    return DCFu/Bdummy

def compute_minDCF(llr, L, working_point):
    t_set=[- np.inf , np.inf]
    
    for s in llr:
        t_set.append(s)
        
    t_set_sorted=np.sort(t_set) 
    DCF_list=[]
        
    for t in t_set_sorted:
        M=Bayes_decision_binary(working_point, llr, L, t)
        DCFu=Bayes_risk(M, working_point)
        DCF=normalize_DCF(DCFu, working_point)
        DCF_list.append(DCF)
    
    minDCF=min(DCF_list)
    
    return minDCF

def compute_DCF_minDCF(llr, LTE, working_point):
    M = Bayes_decision_binary(working_point, llr, LTE)
    
    DCFu = Bayes_risk(M, working_point)

    DCF = normalize_DCF(DCFu, working_point)
    
    minDCF = compute_minDCF(llr, LTE, working_point)

    return M, DCF, minDCF

def Bayes_error_plots(llr, L, model):
    
    effPriorLogOdds = np.linspace(-4, 4, 40)
    
    DCF_vec=[]
    minDCF_vec=[]
    
    
    pi_tilde_vec= 1 / (1+np.exp(-effPriorLogOdds))
    
    for prior in  pi_tilde_vec:
        print("BEP")
        working_point=np.array([prior,1,1])
        M=Bayes_decision_binary(working_point, llr, L)
        DCFu=Bayes_risk(M, working_point)
        DCF=normalize_DCF(DCFu, working_point)
        DCF_vec.append(DCF)
        minDCF=compute_minDCF(llr, L, working_point)
        minDCF_vec.append(minDCF)
    
    

    plt.figure()
    plt.plot(effPriorLogOdds, DCF_vec, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF_vec, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.xlabel('Effective prior log odds')
    plt.title(f'Bayes error plot for {model} :')
    plt.savefig(f'project\plots\BayesErrorPlots\BEP_{model}.pdf')
        
    return DCF_vec, minDCF_vec