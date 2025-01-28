import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
from math import e
from lib import DR, utils, Gaussian_models, DCF, LR, SVM, GMM, Calibration
import itertools

def mcol(v):
    return v.reshape((v.size, 1))

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def try_plot():
    C_values = [0.1, 1, 10, 100, 1000]
    lambda_values = [e**-2, 0.1, 1, 10]
    DCFlist = {
        e**-2: [0.2, 0.18, 0.15, 0.14, 0.13],
        0.1: [0.22, 0.20, 0.17, 0.16, 0.14],
        1: [0.25, 0.23, 0.20, 0.18, 0.16],
        10: [0.3, 0.28, 0.25, 0.22, 0.20]
    }
    minDCFlist = {
        e**-2: [0.18, 0.17, 0.14, 0.13, 0.12],
        0.1: [0.20, 0.19, 0.16, 0.15, 0.13],
        1: [0.23, 0.21, 0.18, 0.17, 0.15],
        10: [0.28, 0.25, 0.22, 0.20, 0.18]
    }
    d = {e**-2:"e-2"}
    plt.figure()

    # Loop over each lambda value to plot the curves
    for lambda_value in lambda_values:
        if lambda_value == e**-2:
            lambda_valuep= "e-2"
        else:
            lambda_valuep=lambda_value
        plt.plot(C_values, DCFlist[lambda_value], label=f'actual DCF (γ={lambda_valuep})', linestyle='-', marker='o')
        plt.plot(C_values, minDCFlist[lambda_value], label=f'min DCF (γ={lambda_valuep})', linestyle='--', marker='x')

    plt.xscale('log', base=10)
    plt.xlabel('C values')
    plt.legend()
    plt.title('DCF and min DCF for different λ values')

    # Add a legend with proper location
    
    plt.show()



if __name__ == '__main__':

    # traj = np.load("project\simulated_trajectory_1.npy")
    # print(traj.shape)
    # print("///")
    # print(traj[0][0][0],traj[1][0][0])
    
    #rows are features and columns are samples
    D, L= utils.load('project/trainData.txt')
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # mu,C=utils.compute_mu_C(D)
    # print(mu.shape,C.shape)
    # exit()
    run = "Calibration"

    if run == "DR":
        print(f"Dataset size: {D.shape[1]}")
        print(f"Number of samples for class 0(Counterfeit): {D0.shape[1]}")
        print(f"Number of aamples for class 1(Genuine): {D1.shape[1]}")
        

        mu =  vcol(D.mean(1))
        print('Mean:')
        print(mu)
        print()

        DC = D - mu
        
        C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
        print('Covariance:')
        print(C)
        print()

        var = D.var(1)
        std = D.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()

        #apply PCA 6
        DP, P = DR.compute_PCA(D, 6)
        #DR.plot_hist(DP, L)

        #apply LDA
        DW, W = DR.compute_LDA(D, L)

        #classify using LDA

        # DTR and LTR are model training data and labels
        # DVAL and LVAL are validation data and labels
        (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
        #print("//////")
        #print(DTR[:, LTR==0].shape[1],DTR[:, LTR==1].shape[1])
        #print(DVAL[:, LVAL==0].shape[1],DVAL[:, LVAL==1].shape[1])
        #try center the dataset!
        #mu = vcol(DTR.mean(1))
        #DTR, DVAL = DTR - mu, DVAL - mu 
        
        #apply PCA
        DTR_pca, P = DR.compute_PCA(DTR, m=2) # best is 4/3 with 0.0925 instead of 0.093
        DVAL_pca = np.dot(P.T, DVAL)

        #DR.classify_LDA(DTR,LTR,DVAL,LVAL)

    #//////////////////
    elif run == "MVG":

        (DTRs, LTR), (DTEs, LTE) = utils.split_db_2to1(D, L)
        #working_points = [[0.5, 1.0, 1.0],[0.9, 1.0, 1.0],[0.1, 1.0, 1.0],[0.5, 1.0, 9.0],[0.5, 9.0, 1.0]]

        #eff_priors = [DCF.compute_effective_prior(working_point) for working_point in working_points]
        # m_vec = [3,4,5]

        # for m in m_vec:
        #     #Gaussian_models.compute_per_class_densities(D, L)
        #     DTR, P = DR.compute_PCA(DTRs, m) 
        #     DTE = np.dot(P.T, DTEs)

        #     working_points = [[0.5, 1.0, 1.0],[0.9, 1.0, 1.0],[0.1, 1.0, 1.0]]

        #     eff_priors = [DCF.compute_effective_prior(working_point) for working_point in working_points]
            
        #     for working_point in working_points:

        #         print(f"results using PCA with m={m} and {working_point[0]} as effective prior for working point {working_point}:")
        #         Gaussian_models.MVG_DCF(DTR, LTR, DTE, LTE, working_point, "MVG", verbose=True)

        #         Gaussian_models.MVG_DCF(DTR, LTR, DTE, LTE, working_point, "Tied", verbose=True)

        #         Gaussian_models.MVG_DCF(DTR, LTR, DTE, LTE, working_point, "Naive", verbose=True)
                
                # Gaussian_models.MVG_classifier(DTR, LTR, DTE, LTE, eff_prior)

                # Gaussian_models.MVG_Tied_classifier(DTR, LTR, DTE, LTE, eff_prior)

                # Gaussian_models.Naive_Bayes_classifier(DTR, LTR, DTE, LTE, eff_prior)
        # best_m = 5
        # DTR, P = DR.compute_PCA(DTRs, best_m) 
        # DTE = np.dot(P.T, DTEs)
        working_point = [0.1, 1.0, 1.0]
        Gaussian_models.MVG_DCF(DTRs, LTR, DTEs, LTE, working_point, "MVG", plot=True)

        Gaussian_models.MVG_DCF(DTRs, LTR, DTEs, LTE, working_point, "Tied", plot=True)

        Gaussian_models.MVG_DCF(DTRs, LTR, DTEs, LTE, working_point, "Naive", plot=True)
    
    elif run=="LR":

        (DTR, LTR), (DTE, LTE) = utils.split_db_2to1(D, L)
        reduced = False
        prior_weighted = False
        quadratic = False
        centered = True
        l_values = np.logspace(-4, 2, 13)
        piT = 0.1
        DCFlist = []
        minDCFlist = []

        if reduced:
            DTR, LTR = DTR[:, ::50], LTR[::50]

        if centered:
            mu, _ = utils.compute_mu_C(DTR)
            DTR = DTR - mu
            DTE = DTE - mu

        for l in l_values:

            DCF_, minDCF = LR.LogisticRegression(DTR,LTR,DTE,LTE,l,piT, prior_weighted, quadratic, centered=centered)
            DCFlist.append(DCF_)
            minDCFlist.append(minDCF)
        
        LR.plot_DCF_lambda(l_values, DCFlist, minDCFlist, prior_weighted=prior_weighted, quadratic=quadratic, centered=centered)
    
    elif run=="SVM":

        (DTR, LTR), (DTE, LTE) = utils.split_db_2to1(D, L)
        K = 1
        #C = 1
        centered=False
        C_values = np.logspace(-5, 0, 11)
        piT = 0.1
        kernelType = "RBF"

        if kernelType == "RBF":
            C_values = np.logspace(-3, 2, 11)

        eps = 1
        c=1
        d=2
        gamma_values = [e**-4,e**-3,e**-2,e**-1]
        DCFlist = {gamma:[] for gamma in gamma_values}
        minDCFlist = {gamma:[] for gamma in gamma_values}
        

        if centered:
            mu, _ = utils.compute_mu_C(DTR)
            DTR = DTR - mu
            DTE = DTE - mu
        C=31.622776601683793
        gamma=0.1353352832366127
        DCF_, minDCF = SVM.SVM(DTR,LTR,DTE,LTE,K,C,piT,kernelType,centered,eps,gamma)
        
        for C in C_values:
            for gamma in gamma_values:
                DCF_, minDCF = SVM.SVM(DTR,LTR,DTE,LTE,K,C,piT,kernelType,centered,eps,gamma)
                DCFlist[gamma].append(DCF_)
                minDCFlist[gamma].append(minDCF)
        
        SVM.plot_DCF_C(C_values,DCFlist,minDCFlist,kernelType,K,centered,gamma_values,gamma_values[0])

    elif run=="GMM":

        (DTR, LTR), (DTE, LTE) = utils.split_db_2to1(D, L)

        piT = 0.1
        wp = [0.1,1,1]
        variant = "Diag"
        n=5
        components_list = [2**i for i in range(n + 1) ]
        DCFlist=[]
        minDCFlist=[]
        comp_c0 = 8
        comp_c1 = 16
        
        #GMM.train_combinations(DTE,LTE,components_list,variant,piT)

        
        folder_path = 'project\saved_models\GMM\GMM_Scores_Diag_piT=0.1_componentsC0=8_componentsC1=16.npy'
        #path = folder_path +'\GMM_Scores_' + str(variant)+ '_piT=' + str(piT) + '_componentsC0='+str(comp_c0)+ '_componentsC1='+str(comp_c1)
        scores = np.load(folder_path)
        model = f"GMM_{variant}_compC0={comp_c0}_compC1={comp_c1}"
        DCF.Bayes_error_plots(scores,LTE,model)

        exit()
        # for components in components_list:
        #     DCF_, minDCF = GMM.GMM(DTR,LTR,DTE,LTE,piT,variant,components)
        #     DCFlist.append(DCF_)
        #     minDCFlist.append(minDCF)

        # GMM.plot_DCF_GMM(components_list,DCFlist,minDCFlist,variant,piT)

    elif run=="Comparison":

        (DTR, LTR), (DTE, LTE) = utils.split_db_2to1(D, L)
        print(DTR.shape,DTE.shape)
        n=3
        print("b")
        scores_LR = np.load("project\saved_models\LR\LR_quadratic_Sllr_l=0.03162277660168379_piT=0.1.npy")
        scores_SVM = np.load("project\saved_models\SVM\SVM_Scores_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")#np.load("project\saved_models\SVM\SVM_Scores_RBF_piT=0.1_K=1_C=100.0_gamma=0.04978706836786395.npy")#
        scores_GMM = np.load("project\saved_models\GMM\GMM_Scores_Diag_piT=0.1_components=8.npy")
        print(scores_LR.shape,scores_SVM.shape,scores_GMM.shape)
        models = ["LR", "SVM", "GMM"]
        scores = [scores_LR, scores_SVM, scores_GMM]
        DCF.Bayes_error_plots(scores[1],LTE,models[1])
        exit()
        
        print("a")
        for i in range(n):

            DCF.Bayes_error_plots(scores[i],LTE,models[i])
    
    elif run == "Calibration":

        DEVAL, LEVAL= utils.load('project/evalData.txt')
        (DTR, LTR), (DVAL, LVAL) = utils.split_db_2to1(D, L)
        fusion = True
        compute_scores_=True
        #load models scores on validation set
        scores_LR = np.load("project\saved_models\LR\LR_quadratic_Sllr_l=0.03162277660168379_piT=0.1.npy")
        scores_SVM = np.load("project\saved_models\SVM\SVM_Scores_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")#np.load("project\saved_models\SVM\SVM_Scores_RBF_piT=0.1_K=1_C=100.0_gamma=0.04978706836786395.npy")#
        #scores_GMM = np.load("project\saved_models\GMM\GMM_Scores_Diag_piT=0.1_components=8.npy")
        #scores = [scores_LR, scores_SVM, scores_GMM]
        models=[]
        test_gmms = True
        components_list = [4,8,16,32]
        variants = ["Full","Diag"]

        models_GMM = [(1,32),(8,32)]
        models = []
        scores_GMM = np.load('project\saved_models\GMM\GMM_Scores_Diag_piT=0.1_componentsC0=8_componentsC1=16.npy')
        model = f"GMM_Diag_compC0=8_compC1=16"
        if test_gmms:
            scores_GMM_eval_list = []
            scores_GMM_list = []
            models = []
            if compute_scores_:
                for (c0,c1) in models_GMM:
                    params_GMM0 = utils.load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm0_Diag_piT=0.1_components={c0}.json")
                    params_GMM1 = utils.load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm1_Diag_piT=0.1_components={c1}.json")
                    gmm = [params_GMM0,params_GMM1]
                    model = f"GMM_Diag_compC0={c0}_compC1={c1}"
                    llrGMM = GMM.compute_gmm_scores(DEVAL,gmm)
                    path = f'project/saved_models/GMM/eval_GMM_Scores_Diag_piT=0.1_componentsC0={c0}_componentsC1={c1}.npy'
                    np.save(path, llrGMM)
                    scores_GMM_eval = np.load(f'project/saved_models/GMM/eval_GMM_Scores_Diag_piT=0.1_componentsC0=8_componentsC1=16.npy')
                    scores_GMM_eval_list.append(scores_GMM_eval)
                    scores_GMM = np.load(f'project\saved_models\GMM\GMM_Scores_Diag_piT=0.1_componentsC0={c0}_componentsC1={c1}.npy')
                    scores_GMM_list.append(scores_GMM)
                    models.append(model)

            scores_l = [(scores_GMM_list[i],scores_GMM_eval_list[i]) for i in range(len(scores_GMM_list))]
            
            systems = [{"model": models[i],"scores": scores, "evalscores":evalscores } for i,(scores,evalscores) in enumerate(scores_l)]
            fusion = []#[0,1,2]
            KFOLD=5
            Calibration.KFold_CrossValidation(systems,LVAL,KFOLD,LEVAL,fusion=fusion,skip_cal=False)

        # scores_LR_eval = np.load("project\saved_models\LR\LR_quadratic_Sllr_eval_l=0.03162277660168379_piT=0.1.npy")
        # scores_SVM_eval = np.load("project\saved_models\SVM\eval_SVM_Scores_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")
        # scores_GMM_eval = np.load("project\saved_models\GMM\eval_GMM_Scores_Diag_piT=0.1_componentsC0=8_componentsC1=16.npy")
        # scores_l=[(scores_LR,scores_LR_eval),(scores_SVM,scores_SVM_eval),(scores_GMM,scores_GMM_eval)]
        
        # scores_l = [(scores_GMM_list[i],scores_GMM_eval_list[i]) for i in range(len(scores_GMM_list))]
        # models=["LR","SVM",model]
        # systems = [{"model": models[i],"scores": scores, "evalscores":evalscores } for i,(scores,evalscores) in enumerate(scores_l)]
        # fusion = [0,1,2]#[0,1,2]
        # KFOLD=5
        # Calibration.KFold_CrossValidation(systems,LVAL,KFOLD,LEVAL,fusion=fusion,skip_cal=True)
        exit()
        #compute models scores on evaluation set
        if compute_scores_:
            if test_gmms:
                scores_GMM_list = []
                scores_GMM_eval_list = []
                
                for variant in variants:
                    for components in components_list:
                        scores_GMM = np.load(f"project\saved_models\GMM\GMM_Scores_{str(variant)}_piT=0.1_components={str(components)}.npy")
                        scores_GMM_list.append(scores_GMM)
                        model= f"GMM_{str(variant)}_components={str(components)}"
                        models.append(model)
                for variant in variants:
                    for components in components_list:
                        params_GMM0 = utils.load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm0_{str(variant)}_piT=0.1_components={str(components)}.json")
                        params_GMM1 = utils.load_gmm(f"project\saved_models\GMM\GMM_parameters_gmm1_{str(variant)}_piT=0.1_components={str(components)}.json")
                        gmm = [params_GMM0,params_GMM1]
                        llrGMM = GMM.compute_gmm_scores(DEVAL,gmm)
                        path = f'project/saved_models/GMM/eval_GMM_Scores_{str(variant)}_piT=0.1_components={str(components)}.npy'
                        np.save(path, llrGMM)
                        scores_GMM_eval = np.load(f'project/saved_models/GMM/eval_GMM_Scores_{str(variant)}_piT=0.1_components={str(components)}.npy')
                        scores_GMM_eval_list.append(scores_GMM_eval)
            else:       
                params_LR_w = np.load("project\saved_models\LR\LR_quadratic_w_l=0.03162277660168379_piT=0.1.npy")
                params_LR_b = np.load("project\saved_models\LR\LR_quadratic_b_l=0.03162277660168379_piT=0.1.npy")
                params_SVM_alpha = np.load("project\saved_models\SVM\SVM_parameters_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")
                params_GMM0 = utils.load_gmm("project\saved_models\GMM\GMM_parameters_gmm0_Diag_piT=0.1_components=8.json")#np.load("project\saved_models\GMM\GMM_parameters_gmm0_Diag_piT=0.1_components=8.npy")
                params_GMM1 = utils.load_gmm("project\saved_models\GMM\GMM_parameters_gmm1_Diag_piT=0.1_components=8.json")
                
                piT = 0.1
                l =0.03162277660168379
                LR.compute_scores(params_LR_w,params_LR_b,DTR,LTR,DEVAL,LEVAL,piT=piT,l=l,quadratic=True)
                scores_LR_eval = np.load("project\saved_models\LR\LR_quadratic_Sllr_eval_l=0.03162277660168379_piT=0.1.npy")

                K=1
                C=31.622776601683793
                gamma=0.1353352832366127
                kernelType="RBF"
                eps = 1
                SVM.compute_scores(params_SVM_alpha,DTR,LTR,DEVAL,K,C,piT,kernelType,False,eps,gamma)
                scores_SVM_eval = np.load("project\saved_models\SVM\eval_SVM_Scores_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")
                
                gmm = [params_GMM0,params_GMM1]
                llrGMM = GMM.compute_gmm_scores(DEVAL,gmm)
                path = 'project/saved_models/GMM/eval_GMM_Scores_Diag_piT=0.1_components=8.npy'
                np.save(path, llrGMM)
                scores_GMM_eval = np.load("project\saved_models\GMM\eval_GMM_Scores_Diag_piT=0.1_components=8.npy")
        else:
            if test_gmms:
                scores_GMM_list = []
                scores_GMM_eval_list = []
                for variant in variants:
                    for components in components_list:
                        scores_GMM = np.load(f"project\saved_models\GMM\GMM_Scores_{str(variant)}_piT=0.1_components={str(components)}.npy")
                        scores_GMM_list.append(scores_GMM)
                        scores_GMM_eval = np.load(f'project/saved_models/GMM/eval_GMM_Scores_{str(variant)}_piT=0.1_components={str(components)}.npy')
                        scores_GMM_eval_list.append(scores_GMM_eval)
                        model= f"GMM_{str(variant)}_components={str(components)}"
                        models.append(model)
            else:
                scores_LR_eval = np.load("project\saved_models\LR\LR_quadratic_Sllr_eval_l=0.03162277660168379_piT=0.1.npy")
                scores_SVM_eval = np.load("project\saved_models\SVM\eval_SVM_Scores_RBF_piT=0.1_K=1_C=31.622776601683793_gamma=0.1353352832366127.npy")
                scores_GMM_eval = np.load("project\saved_models\GMM\eval_GMM_Scores_Diag_piT=0.1_components=8.npy")
        
        if test_gmms:
            scores_l = [(scores_GMM_list[i],scores_GMM_eval_list[i]) for i in range(len(scores_GMM_list))]
        else:
            scores_l = [(scores_LR,scores_LR_eval),(scores_SVM,scores_SVM_eval),(scores_GMM,scores_GMM_eval)] #list of tuple
            models=["LR","SVM","GMM"]

        systems = [{"model": models[i],"scores": scores, "evalscores":evalscores } for i,(scores,evalscores) in enumerate(scores_l)]
        fusion = []#[0,1,2]
        KFOLD=5
        Calibration.KFold_CrossValidation(systems,LVAL,KFOLD,LEVAL,fusion=fusion,skip_cal=False)