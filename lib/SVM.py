import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.optimize
from math import e
from lib.DCF import *
from lib.utils import *


def gradL_hat_dual(alpha, H_):
    n = len(alpha)
    return (np.dot(H_, alpha) - 1).reshape(n)

def L_hat_dual_obj(alpha, H_): # alpha has shape (n,)
        n = len(alpha)
        minusJ_dual = 0.5 * np.dot(np.dot(alpha.T, H_), alpha) - np.dot(alpha.T, np.ones(n)) 
        return minusJ_dual, gradL_hat_dual(alpha, H_)

def L_hat_obj_wrap(H_):

    def L_hat_dual_obj(alpha): # alpha has shape (n,)
        n = len(alpha)
        minusJ_dual = 0.5 * np.dot(np.dot(alpha.T, H_), alpha) - np.dot(alpha.T, np.ones(n)) 
        return minusJ_dual, gradL_hat_dual(alpha, H_)
    
    return L_hat_dual_obj


def L_hat_dual_obj_kern(alpha, H_kern): # alpha has shape (n,)
    n = len(alpha)
    minusJ_dual = 0.5 * np.dot(np.dot(alpha.T, H_kern), alpha) - np.dot(alpha.T, np.ones(n)) 
    return minusJ_dual, gradL_hat_dual_kern(alpha, H_kern)

def L_hat_kern_obj_wrap(H_kern):

    def L_hat_dual_obj_kern(alpha): # alpha has shape (n,)
        n = len(alpha)
        minusJ_dual = 0.5 * np.dot(np.dot(alpha.T, H_kern), alpha) - np.dot(alpha.T, np.ones(n)) 
        return minusJ_dual, gradL_hat_dual_kern(alpha, H_kern)
    
    return L_hat_dual_obj_kern

def gradL_hat_dual_kern(alpha, H_kern):
    n = len(alpha)
    return (np.dot(H_kern, alpha) - 1).reshape(n)

def primal_obj(w_hat_star, C, LTRz, D_):  
    
    return 0.5*np.linalg.norm(w_hat_star)**2+C*np.sum(np.maximum(0,1 - LTRz*(np.dot(w_hat_star.T,D_))))

def duality_gap(D_, H_, LTRz, w_hat_star, alpha_star,C):
    return primal_obj(w_hat_star,C,LTRz,D_) + L_hat_dual_obj(alpha_star, H_)[0]
    
def predict_labels(D_, H_, S, LTE, LTRz, w_hat_star, alpha_star, C, K):
    primal_loss = primal_obj(w_hat_star,C,LTRz,D_)
    dual_loss = L_hat_dual_obj(alpha_star, H_)[0]
    dgap = duality_gap(D_, LTRz,w_hat_star,alpha_star, C)
    predict_labels = np.where(S > 0, 1, 0)
    acc = sum(predict_labels == LTE) / len(predict_labels)
    print('C=%.1f, K=%d,Primal loss: %f, Dual loss: %f, Duality gap: %.9f, Error rate: %.1f%%'%(C,K,primal_loss,dual_loss,dgap,(1-acc)*100))

def kernel(X1, X2, kernelType, *params):
    eps = params[0]
    kernel = 0
    if kernelType == 'Polynomial':
        
        c = params[1]
        d = params[2]
        
        kernel = (np.dot(X1.T, X2) + c)**d
    elif kernelType == 'RBF':
        gamma = params[1]       
        
        x = np.repeat(X1, X2.shape[1], axis=1)
        y = np.tile(X2, X1.shape[1])
        kernel = np.exp(-gamma * np.linalg.norm(x-y, axis=0).reshape(X1.shape[1],X2.shape[1])**2)
    return kernel + eps

def SVM_DCF(S, DTE, LTE, piT):

    working_point = [piT, 1, 1]
    
    #S = np.dot(w.T,DTE)+b

    M = Bayes_decision_binary(working_point, S, LTE)
    
    DCFu = Bayes_risk(M, working_point)

    DCF = normalize_DCF(DCFu,working_point)
    
    minDCF = compute_minDCF(S, LTE, working_point)
    print(f"Results for actual DCF={DCF} and minDCF={minDCF}")

    return DCF, minDCF

def compute_scores(alpha_star,DTR,LTR,DTE,K,C,piT,kernelType,centered,*params):

    folder_path = 'project/saved_models/SVM/eval_'
    LTRz = np.array(([1 if LTR[i]==1 else -1 for i in range(LTR.shape[0])]), dtype=np.float16)
    kern_target = kernel(DTR,DTE,kernelType, *params)
    S = np.sum(np.dot((alpha_star*LTRz).reshape(1,-1), kern_target), axis=0)

    if len(params)>2:
        if centered:
            path = folder_path +'SVM_Scores_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
            np.save(path, S)
            path = folder_path +'SVM_parameters_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
            np.save(path,alpha_star)
        else:
            path = folder_path +'SVM_Scores_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
            np.save(path, S)
            path = folder_path +'SVM_parameters_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
            np.save(path,alpha_star)
    else:
        if centered:
            path = folder_path +'SVM_Scores_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
            np.save(path, S)
            path = folder_path +'SVM_parameters_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
            np.save(path,alpha_star)
        else:
            path = folder_path +'SVM_Scores_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
            np.save(path, S)
            path = folder_path +'SVM_parameters_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
            np.save(path,alpha_star)

def SVM(DTR,LTR,DTE,LTE,K,C,piT,kernelType,centered,*params):

    folder_path = 'project/saved_models/SVM/'
    N = DTR.shape[1]
    F = DTR.shape[0]
    bounds = [(0,C)] * N

    LTRz = np.array(([1 if LTR[i]==1 else -1 for i in range(LTR.shape[0])]), dtype=np.float16)
    LTRz_matrix = LTRz.reshape(-1,1)*LTRz.reshape(1,-1)
    
    if len(params)==2:
        #RBF
        d = {e**-4:"e-4",
             e**-3:"e-3",e**-2:"e-2",e**-1:"e-1"}
        print(f"Results for SVM with kernel type = {kernelType}, piT={piT}, K = {K}, C={C}, gamma={d[params[1]]}") 
    else:
        print(f"Results for SVM with kernel type = {kernelType}, piT={piT}, K = {K}, C={C}") 

    if kernelType is None:
        D_ = np.vstack((DTR, K*np.ones(N)))
        G_ = np.dot(D_.T, D_)
        H_ =  G_ * LTRz_matrix

        svm_obj = L_hat_obj_wrap(H_)
        
        alpha_star , f, _ = scipy.optimize.fmin_l_bfgs_b(func=svm_obj,bounds=bounds,x0=np.zeros(N), factr=1.0)
        
        w_hat_star = np.sum(alpha_star * LTRz * D_, axis=1)
        
        w_star, b_star = w_hat_star[:-1], w_hat_star[-1]
        
        if K>1:
            NT = DTE.shape[1]
            DTE_ = np.vstack((DTE, K*np.ones(NT)))
            b_star = np.dot(w_hat_star.T,DTE_) - np.dot(w_star.T,DTE)
            S = np.dot(w_star.T,DTE)+b_star
        else:
            S = np.dot(w_star.T,DTE)+b_star
        
        DCF, minDCF = SVM_DCF(S,DTE,LTE,piT)

        path = folder_path +'SVM_Scores_' + str(kernelType)+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)
        np.save(path, S)
        path = folder_path +'SVM_parameters_' + str(kernelType)+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)
        np.save(path,w_hat_star)

    else:
        
        kern = kernel(DTR, DTR, kernelType,*params)
        H_kern = LTRz_matrix* kern
        svm_kern_obj = L_hat_kern_obj_wrap(H_kern)

        alpha_star, primal, _ = scipy.optimize.fmin_l_bfgs_b(func=svm_kern_obj, bounds=bounds,x0=np.zeros(N), factr=1.0)
    
        #wc_star = np.sum(m * LTRz * D_, axis=1)
        #w_star, b_star = wc_star[:-1], wc_star[-1]
        ZTR = LTR * 2.0 - 1.0
        kern_target = kernel(DTR,DTE,kernelType, *params)
        S = np.sum(np.dot((alpha_star*LTRz).reshape(1,-1), kern_target), axis=0)
        
        dual_loss = L_hat_dual_obj_kern(alpha_star, H_kern)[0]
        predicted_labels = np.where(S > 0, 1, 0)
        acc_k = sum(predicted_labels == LTE) / len(predicted_labels)
        print('C=%.1f, K=sqrt(eps)=%d, Dual loss: %f, Error rate: %.1f%%'%(C,K,dual_loss,(1-acc_k)*100))

        DCF, minDCF = SVM_DCF(S,DTE,LTE,piT)

        if len(params)>2:
            if centered:
                path = folder_path +'SVM_Scores_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
                np.save(path, S)
                path = folder_path +'SVM_parameters_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
                np.save(path,alpha_star)
            else:
                path = folder_path +'SVM_Scores_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
                np.save(path, S)
                path = folder_path +'SVM_parameters_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_c=" +str(params[1])+"_d="+str(params[2])
                np.save(path,alpha_star)
        else:
            if centered:
                path = folder_path +'SVM_Scores_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
                np.save(path, S)
                path = folder_path +'SVM_parameters_centered_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
                np.save(path,alpha_star)
            else:
                path = folder_path +'SVM_Scores_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
                np.save(path, S)
                path = folder_path +'SVM_parameters_' + kernelType+ '_piT=' + str(piT) + '_K='+str(K)+ "_C=" + str(C)+"_gamma=" +str(params[1])
                np.save(path,alpha_star)

        
    return DCF, minDCF


def plot_DCF_C(C_values, DCFlist, minDCFlist, kernelType,K,centered,gamma_values, *params):

    if kernelType=="RBF":
        d = {e**-4:"e-4",
             e**-3:"e-3",e**-2:"e-2",e**-1:"e-1"}
        plt.figure()

        for gamma in gamma_values:
            gammap=d[gamma]
            plt.plot(C_values, DCFlist[gamma], label=f'actual DCF (γ={gammap})', linestyle='-', marker='o')
            plt.plot(C_values, minDCFlist[gamma], label=f'min DCF (γ={gammap})', linestyle='--', marker='x')
        plt.xscale('log', base=10)
        
        plt.xlabel('C values')
        plt.legend()
    else:
        plt.figure()
        plt.plot(C_values, DCFlist, label='actual DCF',color='r')
        plt.plot(C_values, minDCFlist, label='min DCF',color='b')
        plt.xscale('log', base=10)
        plt.legend(['DCF','min DCF'])
        plt.xlabel('C values')
    
    plt.title(f'actual and min DCF as functions of C in SVM:')
    if kernelType:
        if len(params)>1:
            if centered:
                plt.savefig(f'project\plots\SVM\DCF_minDCF_centered_C_K={K}_kernel={kernelType}_c={params[0]}_d={params[1]}.pdf')
            else:
                plt.savefig(f'project\plots\SVM\DCF_minDCF_C_K={K}_kernel={kernelType}_c={params[0]}_d={params[1]}.pdf')
        else:
            if centered:
                plt.savefig(f'project\plots\SVM\DCF_minDCF_centered_C_K={K}_kernel={kernelType}_gamma={params[0]}.pdf')
            else:
                plt.savefig(f'project\plots\SVM\DCF_minDCF_C_K={K}_kernel={kernelType}_gamma={params[0]}.pdf')
    else:
        if centered:
            plt.savefig(f'project\plots\SVM\DCF_minDCF_centered_C_K={K}.pdf')
        else:
            plt.savefig(f'project\plots\SVM\DCF_minDCF_C_K={K}.pdf')
