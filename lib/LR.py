import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.datasets
from scipy.special import logsumexp
from lib.DCF import *
from lib.utils import *

def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTar = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wNon = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTar # Apply the weights to the loss computations
        loss[ZTR<0] *= wNon

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        G[ZTR > 0] *= wTar # Apply the weights to the gradient computations
        G[ZTR < 0] *= wNon
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    #print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]


#////////////////////////

def prediction_test(LTE, predicted):
    
    pred_result=[]
    
    for i in range(LTE.shape[0]):
        if(LTE[i]==predicted[i]):
            pred_result.append(True)
        else:
            pred_result.append(False)
    return pred_result

def predict_label(w,b,l,DTE,LTE):
    
    S=np.zeros([DTE.shape[1],1])
    LP=np.zeros([DTE.shape[1],1])
    
    w=np.reshape(w,(DTE.shape[0],-1))
    
    for i in range(DTE.shape[1]):
        S[i]=np.dot(w.T,DTE[:,i:i+1])+b
        if S[i]>0:
            LP[i]=1
        else:
            LP[i]=0
    
    pred_result=prediction_test(LTE, LP)
    
    error_rate= pred_result.count(False)/len(pred_result)
    
    print('Error rate with lambda=%f is: %.1f%%'%(l,error_rate*100))


def compute_grads(DTR, LTR, l, w, S, ZTR, piT=None):

    G = -ZTR / (1.0 + np.exp(ZTR * S))
    
    if piT:
        Xi = computeXi(DTR, LTR, ZTR, piT)
        Gx = (vrow(G) * DTR)*Xi
        Jb = np.expand_dims(np.sum(G*Xi, axis=0),axis=0)
        Jw = l*w + np.sum(Gx, axis=1)
    else:
        Gx = (vrow(G) * DTR)
        Jb = np.expand_dims(np.sum(G, axis=0)/DTR.shape[1],axis=0)
        Jw = l*w + (np.sum(Gx, axis=1)/DTR.shape[1])
    #print(Gx.shape)
    
    #print(Jw.shape,Jb.shape)
    return np.concatenate((Jw, Jb))

def computeXi(DTR, LTR, ZTR, piT):
    nT = np.sum(LTR[LTR==1])
    nF = DTR.shape[1] - nT
    argT = piT / nT
    argF = (1-piT)/nF
    Xi = np.where(ZTR == 1, argT, argF )
    return Xi

def logreg_obj_wrap(DTR, LTR, l, piT=None):
    
    def logreg_obj(v):
    # ...
    # Compute and return the objective function value using DTR,LTR, l
    # ...
        w, b = v[0:-1], v[-1]
        
        ZTR = 2 * LTR - 1
        S = (vcol(w).T @ DTR + b).ravel()
        reg_term = l/2*(LA.norm(w))**2
        if piT:
            Xi = computeXi(DTR,LTR,ZTR, piT)
            logs = np.logaddexp(0, -ZTR * S)* Xi
            sum = np.sum(logs,axis=0)
            avg = sum

        else:
            logs = np.logaddexp(0, -ZTR * S)
            sum = np.sum(logs,axis=0)
            avg=sum/DTR.shape[1]
        
        grads = compute_grads(DTR, LTR, l, w, S, ZTR, piT)

        return reg_term+avg, grads
    
    return logreg_obj

def LR_DCF(w, b, DTE, LTE, piT, arg):

    working_point = [piT, 1, 1]
    Sllr=np.zeros([DTE.shape[1]])
    w=np.reshape(w,(DTE.shape[0],-1))
    
    for i in range(DTE.shape[1]):
        Sllr[i]=np.dot(w.T,DTE[:,i:i+1])+b - np.log(arg)
    

    M = Bayes_decision_binary(working_point, Sllr, LTE)
    
    DCFu = Bayes_risk(M, working_point)

    DCF = normalize_DCF(DCFu,working_point)
    
    minDCF = compute_minDCF(Sllr, LTE, working_point)
    
    print(f"Results for actual DCF={DCF} and minDCF={minDCF}")

    return Sllr, DCF, minDCF

def quadratic_expansion(DTR, DTE):
    nT = DTR.shape[1]
    nEv = DTE.shape[1]
    nf_tot = DTR.shape[0] ** 2 + DTR.shape[0]
    quad_dtr = np.zeros((nf_tot, nT))
    quad_dte = np.zeros((nf_tot, nEv))

    for i in range(nT):
        quad_dtr[:, i:i + 1] = compute_expansion(DTR[:, i:i + 1])
    for i in range(nEv):
        quad_dte[:, i:i + 1] = compute_expansion(DTE[:, i:i + 1])

    return quad_dtr, quad_dte

def compute_expansion(x):

    xx_t = np.dot(x, x.T)
    column = np.vstack((xx_t.reshape(-1, 1), x))
    return column

def compute_scores(w,b,DTR,LTR,DTE,LTE,piT,l=None,prior_weighted=False, quadratic=False, centered=False):
    if quadratic:
        DTR,DTE = quadratic_expansion(DTR,DTE)
        
    if prior_weighted:
        
        arg = piT / (1-piT)
    else:
        nT = np.sum(LTR[LTR==1])
        nF = DTR.shape[1] - nT
        arg = nT / nF

    folder_path = 'project/saved_models/LR/'
    print(f"Results for eval LR with , piT={piT} and pw = {prior_weighted} and quadratic={quadratic}") #
    Sllr, DCF, minDCF = LR_DCF(w, b, DTE, LTE, piT, arg)
    path = folder_path +'LR_quadratic_Sllr_eval_' + 'l=' + str(l) + '_piT=' + str(piT)
    np.save(path, Sllr)
    return DCF, minDCF

# quadratic: phi=[vec(x*x^T),x]^T
def LogisticRegression(DTR, LTR, DTE, LTE, l, piT, prior_weighted=False, quadratic=False, centered=False, eval=False):

        
    if quadratic:
        DTR, DTE = quadratic_expansion(DTR,DTE)
        
    if prior_weighted:
        logreg_obj = logreg_obj_wrap(DTR, LTR, l, piT)
        arg = piT / (1-piT)
    else:
        logreg_obj = logreg_obj_wrap(DTR, LTR, l)
        nT = np.sum(LTR[LTR==1])
        nF = DTR.shape[1] - nT
        arg = nT / nF
    
    x0 = np.zeros(DTR.shape[0] + 1)
    
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj,x0,approx_grad=False)

    w, b = x[0:-1], x[-1]
    print(f"Results for LR with l= {l}, piT={piT} and pw = {prior_weighted} and quadratic={quadratic}") #
    Sllr, DCF, minDCF = LR_DCF(w, b, DTE, LTE, piT, arg)

    folder_path = 'project/saved_models/LR/'
    if prior_weighted:
        path = folder_path +'LR_prior_weighted_w_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, w)
        path = folder_path +'LR_prior_weighted_b_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, b)
        path = folder_path +'LR_prior_weighted_Sllr_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, Sllr)
    elif centered:
        path = folder_path +'LR_centered_w_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, w)
        path = folder_path +'LR_centered_b_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, b)
        path = folder_path +'LR_centered_Sllr_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, Sllr)
    elif quadratic:
        
        path = folder_path +'LR_quadratic_w_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, w)
        path = folder_path +'LR_quadratic_b_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, b)
        path = folder_path +'LR_quadratic_Sllr_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, Sllr)
    else:
        path = folder_path +'LR_w_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, w)
        path = folder_path +'LR_b_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, b)
        path = folder_path +'LR_Sllr_' + 'l=' + str(l) + '_piT=' + str(piT)
        np.save(path, Sllr)

    return DCF, minDCF


def plot_DCF_lambda(l_values, DCFlist, minDCFlist, reduced=False, prior_weighted = False, quadratic=False, centered=False):

    plt.figure()
    plt.plot(l_values, DCFlist, label='actual DCF',color='r')
    plt.plot(l_values, minDCFlist, label='min DCF',color='b')
    plt.xscale('log', base=10)
    plt.legend(['DCF','min DCF'])
    plt.xlabel('lambda values')
    
    plt.title(f'actual and min DCF as functions of lambda in LR:')
    if reduced:
        plt.savefig(f'project\plots\LR\DCF_minDCF_l_reduced.pdf')
    elif prior_weighted:
        plt.savefig(f'project\plots\LR\DCF_minDCF_l_pw.pdf')
    elif quadratic:
        plt.savefig(f'project\plots\LR\DCF_minDCF_l_quad.pdf')
    elif centered:
        plt.savefig(f'project\plots\LR\DCF_minDCF_l_centered.pdf')
    else:
        plt.savefig(f'project\plots\LR\DCF_minDCF_l.pdf')
    
    