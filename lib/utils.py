import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
import seaborn as sns
import json
def mcol(v):
    return v.reshape((v.size, 1))

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1, v.size))

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

classes = {0: "Counterfeit",
           1: "Genuine"}

def compute_mu_C(D):

    mu =  vcol(D.mean(1))

    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])

    return mu, C

def plot_Pearson_corr(D, label):
    
    _, C = compute_mu_C(D)
    Corr = C / ( vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5) )
    print(Corr.shape)
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Pearson Correlation Heatmap for class {classes[label]}')
    plt.savefig(f'project\plots\PearsonCorr\Pearson_Corr_{classes[label]}.pdf')
    #plt.show()

def prediction_test(LTE, llr, t):

    predictions = np.where(llr >= t, 1, 0)

    pred_result=[]
    for i in range(LTE.shape[0]):
        if(LTE[i] == predictions[i]):
            pred_result.append(True)
        else:
            pred_result.append(False)
    return pred_result

def diagonalize(mat):
    
    diag=mat*np.identity(mat.shape[0])
    
    return diag

