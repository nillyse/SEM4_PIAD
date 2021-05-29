import numpy as np
import pandas as pd
import scipy.sparse as scsp
import pprint as pr
from sklearn.datasets import fetch_rcv1


zoo = pd.read_csv('zoo.csv')
audytoria = pd.read_csv('audytoria.csv')
print(zoo)
def freq(x, prob=True):
    values = {}
    for el in x:
        if el in values:
            values[el] += 1
        else:
            values[el] = 1
    xi = list(values.keys())
    ni = values.values()
    if(prob):
        pi = {}
        for i,val in enumerate(ni):
            pi[xi[i]] = val/len(x)
        return xi, pi
    return xi, ni


xi, pi = freq(zoo['tail'])
print(xi, pi)

def freq2(X,Y, prob=True):
    if (len(X) != len(Y)):
        return
    values = {}
    for x,y in zip(X, Y):
        if (x in values):
            if(y not in values[x]):
                values[x][y] = {}
                values[x][y] = 1
            else:
                values[x][y] += 1
        else:
            values[x] = {}
            values[x][y] = {}
            values[x][y] = 1
    ni = []
    ni = values.values()
    xi = values.keys()
    if(not prob):
        return xi, ni
    pi = {}
    for n in xi:
        pi[n] = {}
        for x in values[n].keys():
            pi[n][x] = {}
            pi[n][x] = values[n][x]/len(X)
        sum(pi)
    return xi, pi


def entropy(pi):
    entropia = 0
    for i in pi.keys():
        if pi[i] != 0:
            entropia -= pi[i] * np.log2(pi[i])
    return entropia
    
def entropia_warunkowa(pi, pix):
    entropiaTrue = 0
    for i in pi.keys():
        entropia = 0
        for j in pi[i].keys():
            p = pi[i][j]/pix[i]
            entropia -= p * np.log2(p)
        entropiaTrue += pix[i]*entropia
    return entropiaTrue



#def infogain(entropia_x,entropia_w_xy):
#    return entropia_x - entropia_w_xy


def infogain(X, Y):
    x, pix = freq(X)
    xy, pixy = freq2(X, Y)
    entropia = entropy(pix)
    entropia_w = entropia_warunkowa(pixy,pix)
    return entropia - entropia_w



def ID3(data,originaldata,features,target_attribute_name="type",parent_node_class = None):
    
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    
    elif len(features) ==0:
        return parent_node_class

    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        item_values = [infogain(data[feature],data[target_attribute_name]) for feature in features]
        print(item_values)

        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature:{}}
        
        
        features = [i for i in features if i != best_feature]
        
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = ID3(sub_data,originaldata,features,target_attribute_name,parent_node_class)
            
            tree[best_feature][value] = subtree
            
        return(tree)    



features = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
tree = ID3(zoo, zoo, features)
pr.pprint(tree)



rcv1 = fetch_rcv1()
X = rcv1["data"]
Y = rcv1.target[:,87]
X2 = X > 0
Y = Y.toarray().flatten()
print(Y)
info_g = []



for i in range(10):
    X3 = X2[:,i].toarray().flatten()
    info_g.append(infogain(Y, X3))


print(max(info_g))


#test
#xi, pix = freq(audytoria['x'])
#xi, piy = freq(audytoria['y'])
#entropia = entropy(pix)
#entropia = entropy(piy)
#xi, pixy = freq2(audytoria['x'], audytoria['y'])
#xi, piyx = freq2(audytoria['y'], audytoria['x'])

#entropia_w = entropia_warunkowa(pixy, pix)
#entropia_w = entropia_warunkowa(piyx, piy)

#print(infogain(entropia, entropia_w))
#print(infogain(entropia, entropia_w))





            

