from scipy import *
from numpy import *
import sys

#www.ma.utexas.edu/users/zmccoy/nmf.py

def kl_div(A,B):
    "Calculates the KL divergence D(A||B) between the distributions A and B.\nUsage: div = kl_divergence(A,B)"
    D = .0
    # if A.shape is not B.shape: some kind of error management is needed.
    i_stop, j_stop = A.shape
    for i in range(i_stop):
        for j in range(j_stop):
            if A[i,j] != .0:
                D += A[i,j] * math.log( A[i,j] / B[i,j] ) - A[i,j] + B[i,j]
            else:
                D+= B[i,j]
    return D

# The updateStep function is replacable with whatever for whatever distance measure we are using. It takes the matrices V, W, H as argument and returns the updated W, H.
def updateStep(V,W,H):
    "Updates W,H according to the paper by Lee and Seoung, using the KL-div version.\nUsage: W,H = updateStep(V,W,H)"
    Wn = zeros( (W.shape) )
    Hn = zeros( (H.shape) )
    (i_stop, a_stop ) = W.shape
    (a_stop, m_stop ) = H.shape
    WH = dot(W,H)
    # First update W
    for i in range(i_stop):
        for a in range(a_stop):
            Nominator = 0.
            for m in range(m_stop):
                if V[i,m] != .0:
                    Nominator += ( H[a,m] * V[i,m] / WH[i,m] )
                    #if H[a,m] * V[i,m] / WH[i,m] > 5:
                        #sys.stderr.write('\n==============\nPotentially fucked up!\nHV/WM: ' + str(H[a,m] * V[i,m] / WH[i,m])+'\nWH:' + str(WH[i,m])+'\nV:' + str(V[i,m]) + '\nH:' + str(H[a,m]) + '\n')
            Wn[i,a] = ( W[i,a] * Nominator / sum(H,1)[a] )
    # Then update H
    WH = dot(Wn,H)
    W = Wn
    for a in range(a_stop):
        for m in range(m_stop):
            Nominator = 0.
            for i in range(i_stop):
                if V[i,m] != .0:
                    Nominator += ( W[i,a] * V[i,m] / WH[i,m] )
            Hn[a,m] = ( H[a,m] * Nominator / sum(W,0)[a] )
    return Wn,Hn


# V = scipy.io.read_array('Personality.csv',columns = (8,-1), lines= (1,-1) )


# this should not take a number of reps, it should take a greatest allowable KL-distance
# V is data matrix, r the desired rank and reps is the number of repetitions.
def nmf(V,r,thresh=0.9999,W=False,H=False):
    "Does nmf on data. nmf(V,r,iterations), where V is the data matrix, r the desired rank, and iterations is the number of iterations. Uses up a lot of memory, might be better ways ...\nUsage: W,H = nmf(V,r,iterations)"
    import time
    oldKL = 9999999999999.
    (i,m) = V.shape
    if W is False:
                W = random.rand(i,r)
                H = random.rand(r,m)
    start_time = time.time()
    prev_time = start_time
    p = True
    step = 0
    while p:
#    for i in range(iterations):
        step += 1
        W,H = updateStep(V,W,H)
        KL = kl_div(V,dot(W,H))
        if KL >= thresh * oldKL:
                        p = False
        oldKL = KL
        this_time = time.time()
        elapsed_time = this_time -start_time
        loop_time = this_time - prev_time
        sys.stderr.write('\n----------------------------\nUpdate step ' + str(step) + ' finished in ' + str(round(loop_time,2)) + 's.\nTime elapsed is ' + str(round(elapsed_time,2)) + 's. \n' + 'KL divergence is ' + str(KL) + '\n----------------------------\n')
        #sys.stderr.write(str(H)+'\n') # shows H every update.
        prev_time = this_time
    return W,H


# pylab.imshow(W,origin='lower',aspect=.01,interpolation='nearest',cmap=pylab.cm.summer) shows the picture nicely.

def coclustering(W):
    "Returns a matrix with a one if two cells are in the same cluster, zero otherwise."
    m,r = W.shape
    Q = zeros((m,m))
    clusters = []
    for strain in W:
        clusters.append( strain.argmax() )
    for i in range(m):
        for j in range(i,m):
            try:
                if clusters[i] == clusters[j]:
                    Q[i,j] = 1.0
            except:
                print i,j
                raise
    Q = Q+Q.T - eye(m)
    return Q

def masterRun(files):
    import cPickle
    f = open(files[0],'r')
    (W,H) = cPickle.load(f)
    f.close()
    m,r = W.shape
    Q = zeros( (m,m) )
    for file in files:
        f = open(file,'r')
        (W,H) = cPickle.load(f)
        f.close()
        Q = Q + coclustering(W)
    Q = Q / float( len(files) )
    return Q



# This just finds the labels with the highest load for the different bases.
def findHighs(H,labels):
    "Returns a list of dictionarys with how high different labels load to different factors. Each dictionary is given with loading:label, so it can be sorted by loadings if needed."
    data = []
    for basis in H:
        feature = {}
        for i in range(len(basis)):
            feature[basis[i]] = labels[i]
        data.append(feature)
    return data


def basisType( valueDict, numRows ):
    "Sorts and returns a numRows of the highest loaded labels from findHighs."
    keys = valueDict.keys()
    keys.sort()
    keys.reverse()
    ret = []
    for i in range(numRows):
        ret.append(valueDict[keys[i]])
    return ret