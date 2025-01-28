import numpy as np
from lib.DCF import *
import lib.LR as LR
from lib.utils import * 

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do not need to shuffle scores in this case, but it may be necessary if samples are sorted in peculiar ways to ensure that validation and calibration sets are independent and with similar characteristics   
def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def calibrate_system(KFOLD, scores, labels, eval_scores, eval_labels, pT, pW, model=""):

    #axes = np.array([ [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [None, plt.figure().gca(), plt.figure().gca()] ])
    calibrated_scores = [] # We will add to the list the scores computed for each fold
    labels_pool = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.

    # logOdds, actDCF, minDCF = bayesPlot(scores, labels)
    # axes[0,0].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
    # axes[0,0].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF')

    # logOdds, actDCF, minDCF = bayesPlot(scores, labels)
    # axes[1,0].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'minDCF')
    # axes[1,0].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'actDCF')
    
    # axes[0,0].set_ylim(0, 0.8)    
    # axes[0,0].legend()

    # axes[1,0].set_ylim(0, 0.8)    
    # axes[1,0].legend()
    
    # axes[0,0].set_title('System 1 - validation - non-calibrated scores')
    # axes[1,0].set_title('System 2 - validation - non-calibrated scores')
    # # We plot the non-calibrated minDCF and actDCF for reference
    logOdds, actDCF_precal, minDCF_precal = bayesPlot(scores, labels)
    # axes[0,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF (pre-cal.)')
    # axes[0,1].plot(logOdds, actDCF, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
    print ('System 1')
    print ('\tValidation set')
    print ('\t\tminDCF(p=0.1), no cal.: %.3f' % compute_minDCF_binary_fast(scores, labels, pT, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(scores, labels, pT, 1.0, 1.0))
    
    # train the calibration model for the prior pT = 0.1
    
    # Train KFOLD times the calibration model
    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores, foldIdx,KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx,KFOLD)
        # Train the model on the KFOLD - 1 training folds
        w, b = LR.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pW)
        # Apply the model to the validation fold
        calibrated_SVAL =  (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the cores list
        calibrated_scores.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_pool.append(LVAL)

    # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)
    calibrated_scores = np.hstack(calibrated_scores)
    labels_pool = np.hstack(labels_pool)

    # Evaluate the performance on pooled scores - we need to use the label vector labels_pool since it's aligned to calibrated_scores   
    print ('\t\tminDCF(p=0.1), cal.   : %.3f' % compute_minDCF_binary_fast(calibrated_scores, labels_pool, pT, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
    print ('\t\tactDCF(p=0.1), cal.   : %.3f' % compute_actDCF_binary_fast(calibrated_scores, labels_pool, pT, 1.0, 1.0))
    
    logOdds, actDCF_cal, minDCF = bayesPlot(calibrated_scores, labels_pool)
    # axes[0,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'actDCF (cal.)') # NOTE: actDCF of the calibrated pooled scores MAY be lower than the global minDCF we computed earlier, since ache fold is calibrated on its own (thus it's as if we were estimating a possibly different threshold for each fold, whereas minDCF employs a single threshold for all scores)
    # axes[0,1].legend()

    # axes[0,1].set_title('System 1 - validation')
    # axes[0,1].set_ylim(0, 0.8)    
    BEP_cal(logOdds,actDCF_precal,actDCF_cal,minDCF_precal,eval=False, model=model)
    # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)

    w, b = LR.trainWeightedLogRegBinary(vrow(scores), labels, 0, pW)

    if len(eval_scores):
        # We can use the trained model for application / evaluation data
        calibrated_eval_scores = (w.T @ vrow(eval_scores) + b - np.log(pT / (1-pT))).ravel()

        print ('\tEvaluation set')
        print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(eval_scores, eval_labels, 0.2, 1.0, 1.0))
        print ('\t\tactDCF(p=0.1), no cal.: %.3f' % compute_actDCF_binary_fast(eval_scores, eval_labels, 0.2, 1.0, 1.0))
        print ('\t\tactDCF(p=0.1), cal.   : %.3f' % compute_actDCF_binary_fast(calibrated_eval_scores, eval_labels, 0.2, 1.0, 1.0))    
        
        # We plot minDCF, non-calibrated DCF and calibrated DCF for system 1
        logOdds, actDCF_precal, minDCF = bayesPlot(eval_scores, eval_labels)
        logOdds, actDCF_cal, _ = bayesPlot(calibrated_eval_scores, eval_labels) # minDCF is the same
        # axes[0,2].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'minDCF')
        # axes[0,2].plot(logOdds, actDCF_precal, color='C0', linestyle=':', label = 'actDCF (pre-cal.)')
        # axes[0,2].plot(logOdds, actDCF_cal, color='C0', linestyle='-', label = 'actDCF (cal.)')
        # axes[0,2].set_ylim(0.0, 0.8)
        # axes[0,2].set_title('System 1 - evaluation')
        # axes[0,2].legend()
        BEP_cal(logOdds,actDCF_precal,actDCF_cal,minDCF,eval=True, model=model)

#systems is a list of dictionaries describing the models to be calibrated
#if fusion is set, it's a list of index of elements in "systems" whose scores need to be fused

def KFold_CrossValidation(systems,  LCAL, KFOLD,  LEVAL=None, fusion=None,skip_cal=False,pT = 0.1,pW=0.1):

    # SAMEFIGPLOTS = True # set to False to have 1 figure per plot
    
    
    # if SAMEFIGPLOTS:
    #     fig = plt.figure(figsize=(16,9))
    #     axes = fig.subplots(3,3, sharex='all')
    #     fig.suptitle('K-fold')
    # else:
    #     axes = np.array([ [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [plt.figure().gca(), plt.figure().gca(), plt.figure().gca()], [None, plt.figure().gca(), plt.figure().gca()] ])

    print()
    print('*** K-FOLD ***')
    print()

    # Calibrate all systems (independently)
    if not skip_cal:
        for i,system in enumerate(systems):
            minDCF_nocal = compute_minDCF_binary_fast(system["scores"], LCAL, pT, 1.0, 1.0)
            actDCF_nocal = compute_actDCF_binary_fast(system["scores"], LCAL, pT, 1.0, 1.0)

            print('Sys %d (%s) no cal: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (i,
                system["model"],                                                             
                compute_minDCF_binary_fast(system["scores"], LCAL, pT, 1.0, 1.0),
                compute_actDCF_binary_fast(system["scores"], LCAL, pT, 1.0, 1.0)))
    

            calibrate_system(KFOLD,system["scores"],LCAL,system["evalscores"],LEVAL,pT,pW, system["model"])


    #Fusion
    if len(fusion):
        
        systems_sel = [systems[i] for i in fusion]
        
        fusedScores = [] # We will add to the list the scores computed for each fold
        fusedLabels = [] # We need to ensure that we keep the labels aligned with the scores. The simplest thing to do is to just extract each fold label and pool all the fold labels together in the same order as we pool the corresponding scores.
        
        
        
        # Train KFOLD times the fusion model
        for foldIdx in range(KFOLD):
            # keep 1 fold for validation, use the remaining ones for training  
            SCAL_ = []
            SVAL_ = []
            
            for system in systems_sel:

                SCALi, SVALi = extract_train_val_folds_from_ary(system["scores"], foldIdx,KFOLD)
                SCAL_.append(SCALi)
                SVAL_.append(SVALi)
            
            LCAL_, LEVAL_ = extract_train_val_folds_from_ary(LCAL, foldIdx,KFOLD)
            # Build the training scores "feature" matrix
            SCAL_ = np.vstack(SCAL_)
            # Train the model on the KFOLD - 1 training folds
            w, b = LR.trainWeightedLogRegBinary(SCAL_, LCAL_, 0, pT)
            # Build the validation scores "feature" matrix
            SVAL_ = np.vstack(SVAL_)
            # Apply the model to the validation fold
            calibrated_SVAL =  (w.T @ SVAL_ + b - np.log(pT / (1-pT))).ravel()
            # Add the scores of this validation fold to the cores list
            fusedScores.append(calibrated_SVAL)
            # Add the corresponding labels to preserve alignment between scores and labels
            fusedLabels.append(LEVAL_)

        # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
        fusedScores = np.hstack(fusedScores)
        fusedLabels = np.hstack(fusedLabels)

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

        print ('Fusion')
        print ('\tValidation set')
        print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(fusedScores, fusedLabels, pT, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
        print ('\t\tactDCF(p=0.1)         : %.3f' % compute_actDCF_binary_fast(fusedScores, fusedLabels, pT, 1.0, 1.0))

        logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
        BEP_cal(logOdds,None,actDCF,minDCF,eval=False, model="fusion_all_comp")
        # As comparison, we select calibrated models trained with prior 0.2 (our target application)
        # logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_1, labels_sys_1)
        # axes[2,1].set_title('Fusion - validation')
        # axes[2,1].plot(logOdds, minDCF, color='C0', linestyle='--', label = 'S1 - minDCF')
        # axes[2,1].plot(logOdds, actDCF, color='C0', linestyle='-', label = 'S1 - actDCF')
        # logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_sys_2, labels_sys_2)
        # axes[2,1].plot(logOdds, minDCF, color='C1', linestyle='--', label = 'S2 - minDCF')
        # axes[2,1].plot(logOdds, actDCF, color='C1', linestyle='-', label = 'S2 - actDCF')    
        
        # logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
        # axes[2,1].plot(logOdds, minDCF, color='C2', linestyle='--', label = 'S1 + S2 - KFold - minDCF(0.2)')
        # axes[2,1].plot(logOdds, actDCF, color='C2', linestyle='-', label = 'S1 + S2 - KFold - actDCF(0.2)')
        # axes[2,1].set_ylim(0.0, 0.8)
        # axes[2,1].legend()

        # For K-fold the final model is a new model re-trained over the whole set, using the optimal hyperparameters we selected during the k-fold procedure (in this case we have no hyperparameter, so we simply train a new model on the whole dataset)
        S_systems = [systems[i]["scores"] for i in fusion]
        SMatrix = np.vstack(S_systems)
        print(SMatrix.shape)
        w, b = LR.trainWeightedLogRegBinary(SMatrix, LCAL, 0, pT)

        # Apply model to application / evaluation data
        S_systems_eval = [systems[i]["evalscores"] for i in fusion]
        SMatrixEval = np.vstack(S_systems_eval)
        fused_eval_scores = (w.T @ SMatrixEval + b - np.log(pT / (1-pT))).ravel()

        print ('\tEvaluation set')
        print ('\t\tminDCF(p=0.1)         : %.3f' % compute_minDCF_binary_fast(fused_eval_scores, LEVAL, pT, 1.0, 1.0))
        print ('\t\tactDCF(p=0.1)         : %.3f' % compute_actDCF_binary_fast(fused_eval_scores, LEVAL, pT, 1.0, 1.0))
        
        logOdds, actDCF, minDCF = bayesPlot(fused_eval_scores, LEVAL) # minDCF is the same
        BEP_cal(logOdds,None,actDCF,minDCF,eval=True, model="fusion_all_comp")
        



