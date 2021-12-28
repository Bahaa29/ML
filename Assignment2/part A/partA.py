#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# ### read data

# In[14]:


df = np.genfromtxt('house-votes-84.data.txt',dtype='str',delimiter=',')
Copy_data = df
df


# ### majority build

# In[15]:


for i in Copy_data:
    yes = i[i=='y'].size
    no = i[i=='n'].size
    if(yes<=no):
        i[i=='?'] = 'n'
    else:
        i[i=='?'] = 'y'


# In[16]:


Copy_data.shape


# ### spiltdata random by percent 

# In[17]:


def split(data, percent):
    rng = np.random.default_rng()
    xtrain = np.random.choice(data.shape[0], int(data.shape[0] * (percent / 100)),replace=False)
    train = data[xtrain]
    test = np.delete(data, xtrain, axis=0)
    return train,test


# ## Build the tree

# In[18]:


class Nodetree:
    def __init__(self):
        
        self.leftN = None
        self.rightN = None
        self.data = None
        self.midN = None
        
        self.bestCol = None
        self.InformationGain = None
        
        self.EntropyYes = None
        self.EntropyNo = None

        self.ZeroE = None
        self.col = None
        self.remCols = None
            
def recursiveSize(node):
    if node is None:
          return 0
    return 1 + recursiveSize(node.leftN) + recursiveSize(node.rightN)

def Size(node):
    if node is None:
        return 0
    return 1 + recursiveSize(node.leftN) + recursiveSize(node.midN) + recursiveSize(node.rightN) 


# In[19]:


def BestIG(data,Names):
    
    ListGain = []
    k = data.shape[1]
    lebal = data.T[0]
    
    Entropy = lebal[lebal=='republican'].size/lebal.size
    ETarget = -1*( (Entropy)*np.log2(Entropy) + (1-Entropy)*np.log2(1-Entropy) )
    
    for i in range(1,k):
##first
        if(data.T[i][data.T[i]=='y'].size == 0):
            half = 0
        else:
            temp = np.logical_and(data.T[i]=='y', data.T[0]=='republican')
            half = data.T[i][temp].size/data.T[i][data.T[i]=='y'].size

        half2 = 1-half
        EYes = -1*((half)*np.log2(half) + (half2)*np.log2(half2))
        
        if (half == 0 or half == 1):
            EYes = 0

##second
        if (data.T[i][data.T[i]=='n'].size == 0):
            halfn = 0
        else:
            temp2 = np.logical_and(data.T[i]=='n', data.T[0]=='republican')
            halfn = data.T[i][temp2].size/data.T[i][data.T[i]=='n'].size
            
        halfn2 = 1-halfn
        ENo = -1*((halfn)*np.log2(halfn) + (halfn2)*np.log2(halfn2))
        
        if (halfn == 0 or halfn == 1):
            ENo = 0

##thrid
        if (data.T[i][data.T[i]=='?'].size == 0):
            halfm = 0
        else:
            temp3 = np.logical_and(data.T[i]=='?', data.T[0]=='republican')
            halfm = data.T[i][temp3].size/data.T[i][data.T[i]=='?'].size
            
        halfm2 = 1-halfm
        EQ = -1*((halfm)*np.log2(halfm) + (halfm2)*np.log2(halfm2))
        if (halfm == 0 or halfm == 1):
            EQ = 0

        #Calculate Information Gain
        ListGain.append(ETarget- ( (data.T[i][data.T[i]=='y'].size/data.T[i].size)*EYes + (data.T[i][data.T[i]=='n'].size/data.T[i].size)*ENo + (data.T[i][data.T[i]=='?'].size/data.T[i].size)*EQ))
    
    Selected = Nodetree()
    if(EYes == 0):
        leaf = Nodetree()
        leaf.ZeroE = 'republican' if half>half2 else 'democrat'
        Selected.rightN = leaf
    if(ENo == 0):
        leafn = Nodetree()
        leafn.ZeroE = 'republican' if halfn>halfn2 else 'democrat'
        Selected.leftN  = leafn
    if(EQ == 0):
        leafm = Nodetree()
        leafm.ZeroE = 'republican' if halfm>halfm2 else 'democrat'
        Selected.midN = leafm
        
    BIndex = ListGain.index(max(ListGain))
    Selected.col = Names[BIndex]
    Selected.remCols = Names[:BIndex]+Names[BIndex+1:]
    Selected.BIndex = BIndex + 1
    Selected.InformationGain = ListGain[BIndex]
    Selected.data = data
    Selected.EntropyYes = EYes
    Selected.EntropyNo = ENo
    Selected.EntropyQ = EQ

    return Selected
    


# In[20]:


def build(rootNode):
    
    if(rootNode.data.shape[1] == 2):
        
        #right node yes
        temp = np.logical_and(data.T[1]=='y', data.T[0]=='republican')
        if(data.T[1][data.T[1]=='y'].size == 0):
            div1 = 0
        else:
            div1 = data.T[1][temp].size/data.T[1][data.T[1]=='y'].size
        leaf = Nodetree()
        leaf.ZeroE = 'republican' if div1>0.5 else 'democrat'
        rootNode.rightN = leaf

        #left node no
        temp2 = np.logical_and(data.T[1]=='n', data.T[0]=='republican')
        
        if (data.T[1][data.T[1]=='n'].size == 0):
            div2 = 0
        else:
            div2 = data.T[1][temp2].size/data.T[1][data.T[1]=='n'].size

        leaf2 = Nodetree()
        leaf2.ZeroE = 'republican' if div3>0.5 else 'democrat'
        rootNode.leftN = leaf2

        #mid node ?
        temp3 = np.logical_and(data.T[1]=='?', data.T[0]=='republican')
        
        if (data.T[1][data.T[1]=='?'].size == 0):
            div3 = 0
        else:
            div3 = data.T[1][temp3].size/data.T[1][data.T[1]=='?'].size
        
        leaf3 = Nodetree()
        leaf3.ZeroE = 'republican' if div5>0.5 else 'democrat'
        rootNode.midN = leaf3
        
    else:
        
        if(rootNode.rightN == None):
            new = np.copy(rootNode.data)
            new = new[new[:,rootNode.BIndex] == 'y']
            new = np.concatenate((new[:,:rootNode.BIndex], new[:,rootNode.BIndex+1:]), axis = 1)
            if new.shape[0] == 0 :
                leaf = Nodetree()
                leaf.ZeroE = rootNode.data[0][0]
                rootNode.rightN = leaf
            else:
                rootNode.rightN = build(BestIG(new,rootNode.remCols))
        
        
        if(rootNode.leftN == None):
            new = np.copy(rootNode.data)
            new = new[new[:,rootNode.BIndex] == 'n']
            new = np.concatenate((new[:,:rootNode.BIndex], new[:,rootNode.BIndex+1:]), axis = 1)
            if new.shape[0] == 0 :
                leaf = Nodetree()
                leaf.ZeroE = rootNode.data[0][0]
                rootNode.leftN = leaf
            else:
                rootNode.leftN = build(BestIG(new,rootNode.remCols))

        if(rootNode.midN == None):
            new = np.copy(rootNode.data)
            new = new[new[:,rootNode.BIndex] == '?']
            new = np.concatenate((new[:,:rootNode.BIndex], new[:,rootNode.BIndex+1:]), axis = 1)
            if new.shape[0] == 0 :
                leaf = Nodetree()
                leaf.ZeroE = rootNode.data[0][0]
                rootNode.midN = leaf
            else:
                rootNode.midN =build(BestIG(new,rootNode.remCols))

    return rootNode

def predictoutput(rootNode, row):
    if rootNode.ZeroE != None:
        return rootNode.ZeroE
    else:
        if row[rootNode.BIndex] == 'y':
            return predictoutput(rootNode.rightN, np.concatenate((row[:rootNode.BIndex],row[rootNode.BIndex+1:])))
        elif row[rootNode.BIndex] == 'n':
            return predictoutput(rootNode.leftN, np.concatenate((row[:rootNode.BIndex],row[rootNode.BIndex+1:])))
        elif row[rootNode.BIndex] == '?':
            return predictoutput(rootNode.midN, np.concatenate((row[:rootNode.BIndex],row[rootNode.BIndex+1:])))
def pTest(rootNode,testData):
    correct = 0
    k = 0
    for row in testData:
        correct = correct + (1 if predictoutput(rootNode,row) == testData[k][0] else 0)
        k += 1
    return ((correct/testData.shape[0])*100)


# ## for point one 

# In[39]:


print("for RATIO 25", '\n')
for i in range(5):
    train, test = split(df,25)
    colNames = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16']
    root = BestIG(train,colNames)
    rootNode = build(root)
    print('Itr', i+1, ':')
    print('accuracy :{}'.format(pTest(rootNode, test)))
    print('tree size :{}'.format(Size(rootNode)) , '\n')
print('------------------------------------------------')
    


# ## for point two 

# In[36]:


sizes = [30, 40, 50, 60, 70]
length= len(sizes)
for k in range (length):
    accuracies = []
    tSizes = []
    for i in range(5):
        train, test = split(df,sizes[k])
        colNames = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16']
        root = BestIG(train,colNames)
        rootNode = build(root)
        accuracies.append(pTest(rootNode, test))
        tSizes.append(Size(rootNode))
    print('ratio', sizes[k], ':')
    print('Accuracie{}'.format(accuracies)) 
    print('Tree size :',tSizes)
    print("\nMin accuracy: " + str(np.min(accuracies)) +"\nMax accuracy: " + str(np.max(accuracies)) +
              "\nMean accuracy: " + str(np.mean(accuracies)))
    print("Min Tree size: " + str(np.min(tSizes)) +"\nMax Tree size: " + str(np.max(tSizes)) +
              "\nMean Tree size: " + str(np.mean(tSizes)))
    print('-----------------------------')


# In[ ]:





# In[ ]:




