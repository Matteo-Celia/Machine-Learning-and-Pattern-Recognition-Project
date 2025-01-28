import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg

def mcol(v):
    return v.reshape((v.size, 1))

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def plot_hist(D, L):

    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel("LDA direction %d" % dIdx)
        plt.hist(D0[dIdx,:], bins = 10, density = True, alpha = 0.4, label = 'Counterfeit')
        plt.hist(D1[dIdx,:], bins = 10, density = True, alpha = 0.4, label = 'Genuine')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('project\plots\hist_LDA_val_%d.pdf' % dIdx)
    #plt.show()

def plot_hist_v(D, L):

    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    dIdx=0
    
    plt.figure()
    plt.xlabel("LDA direction %d" % dIdx)
    plt.hist(D0[dIdx,:], bins = 10, density = True, alpha = 0.4, label = 'Counterfeit')
    plt.hist(D1[dIdx,:], bins = 10, density = True, alpha = 0.4, label = 'Genuine')
    
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    #plt.savefig('project\plots\hist_LDA_validation_%d.pdf' % dIdx)
    plt.show()

def compute_PCA(D,m):

    assert m <= D.shape[0]
        
    mu =  vcol(D.mean(1))

    DC = D - mu
    
    C = (DC @ DC.T) / float(D.shape[1])

    #compute SVD 
    U, s, Vh = np.linalg.svd(C)

    #retrieve the m largest eigenvectors
    P = U[:, 0:m]
    
    #P[:,1]=-P[:,1]
    # print("P: ")
    # print(P)

    #returns the projected dataset and the projection matrix
    DP = np.dot(P.T, D)

    return DP, P

def classify_PCA(m, DTR, LTR, DVAL, LVAL):

    _, P = compute_PCA(DTR, m)

    #Project samples on the first PCA direction
    DTR_pca = vrow(np.dot(P[:,0].T, DTR))
    DVAL_pca = vrow(np.dot(P[:,0].T, DVAL)) 

    #compute the classifier's threshold using the mean for the two classes
    threshold = (DTR_pca[0, LTR==0].mean() + DTR_pca[0, LTR==1].mean()) / 2.0 

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_pca[0] >= threshold] = 1
    PVAL[DVAL_pca[0] < threshold] = 0

    errors = np.sum(PVAL != LVAL)
    
    plot_hist(DTR_pca,LTR)

    #print("Number of errors(PCA) : ", errors)

def compute_covariance_matrices(D,L):
    
    #select the samples for each class
   
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    #Compute the mean for the class samples
    
    mu0= vcol(D0.mean(1))
    mu1= vcol(D1.mean(1))

    #Remove the mean from the class data
    
    DC0 = D0 - mu0
    DC1 = D1 - mu1

    #compute the covariance matrix for each class (the division is not performe since this will 
    # then be simplified when weighting the sum)
    
    Sw0 = ((DC0) @ (DC0).T) #/ float(DC1.shape[1])
    Sw1 = ((DC1) @ (DC1).T) #/ float(DC2.shape[1])

    #compute within class covariance matrix can be computed as a weighted sum the covariance matrices of each class
    SW = (Sw0 + Sw1)/ float(D.shape[1])

    #compute the between class covariance matrix
    mu = vcol(D.mean(1))
    
    sb0 = ((mu0 - mu) @ (mu0 - mu).T)*float(DC0.shape[1]) 
    sb1 = ((mu1 - mu) @ (mu1 - mu).T)*float(DC1.shape[1]) 

    SB = (sb0 + sb1) / float(D.shape[1])

    return SW, SB

def compute_LDA(D, L):

    SW, SB = compute_covariance_matrices(D, L)

    m = 1 # C - 1 LDA directions so 2 - 1 = 1
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    #if we want to find a basis U for the subspace spanned by W
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m] 

    #returns the projected dataset and the projection matrix
    #W = -W
    DW = np.dot(W.T, D)

    return DW, W

def classify_LDA(DTR, LTR, DVAL, LVAL):

    #apply LDA to both the training set and the validation set
    DTR_lda, W = compute_LDA(DTR, LTR)
    DVAL_lda = np.dot(W.T, DVAL)
    #plot_hist_v(DVAL_lda,LVAL)
    #Projected samples have only 1 dimension (2 classes)
    Fmu = DTR_lda[0, LTR==0].mean()
    Tmu = DTR_lda[0, LTR==1].mean()
    print(Tmu,Fmu)
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0 
    #threshold = 0.021 # 0.021
    print(threshold)
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    errors = np.sum(PVAL != LVAL)

    err_rate = errors/ float(LVAL.shape[0])
    print("Error rate: ", err_rate)

    #plot_ER_LDA(DVAL_lda, LVAL)


def plot_ER_LDA(DVAL_lda, LVAL):

    
    thresholds = np.linspace(-1, 1, 100)  # Generate 100 thresholds from -1 to 1

    #function to compute error rate given a threshold
    def compute_error_rate(threshold):
        
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0

        errors = np.sum(PVAL != LVAL)
        err_rate = errors/ float(LVAL.shape[0])

        return err_rate

    # Compute error rates for each threshold
    error_rates = [compute_error_rate(threshold) for threshold in thresholds]
    min_er = min(error_rates)
    min_threshold = thresholds[error_rates.index(min(error_rates))]
    print(f"min error rate is: {min_er} at threshold {min_threshold}")
    # Plot the error rates
    plt.plot(thresholds, error_rates)
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Threshold')
    plt.grid(True)
    plt.savefig('project\plots\DR\ErrorRate_threshold.pdf')
    #plt.show()