from NNfunction import *
import pandas as pd
import matplotlib.pyplot as plt

raw_df=pd.read_csv("traindata.data") # read data from .data
filtered_df=raw_df[raw_df['at10'] == 2]
allData_df=filtered_df.dropna()  # drop rows with NaN values
allData=allData_df.values.astype(float) # set values type
Wa=([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1]) #[w1a,w2a,w3a,w4a,w5a,w6a,w7a,w8a,w9a,Bias]
W11=([0.1,1]) # [wa11,Bias]
l=-0.9 # learning rate
e11=1  # node 11 error
count=1
errArr=[]
d11=1 

for i in range(len(allData)):
    curr=allData[i]
    X=list(curr[1:10])+[1.0] # input values

    print("\nEpoch:", count)
    print("\n-------------------------Forward pass-------------------------> ")

    # node a
    va=Nout(X,Wa)
    ya=sigmoid(va)
    print("\nSum(V) of node 'a' is: %8.3f, Y from node 'a' is: %8.3f" % (va,ya))

    # node 11
    X11=[ya,1]
    v11=Nout(X11,W11)
    y11=sigmoid(v11)
    print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (v11,y11))

    print("\n<---- Back propagation & calculate new Weights and Biases ----")
    print("\nError 11:",e11)
    errArr.append(e11)

    # node 11
    e11=d11-y11
    g11=gradOut(e11,y11)
    dwa11=deltaw(l,g11,ya)
    wa11n = W11[0]+dwa11
    db11 = deltaw(l,g11,1)
    b11n = W11[1]+db11
    W11[0]=wa11n
    W11[1]=b11n
    print("\nNew wa11 is %8.3f, New bias 11 is %8.3f\n"% (wa11n,b11n))

    # node a
    sumN11w = g11*(W11[0])
    ga = gradH(ya,sumN11w)
    # from 1 into a
    dw1a = deltaw(l,ga,X[0])
    w1an = Wa[0]+dw1a
    # from 2 into a
    dw2a = deltaw(l,ga,X[1])
    w2an = Wa[1]+dw2a    
    # from 3 into a
    dw3a = deltaw(l,ga,X[2])
    w3an = Wa[2]+dw3a 
    # from 4 into a
    dw4a = deltaw(l,ga,X[3])
    w4an = Wa[3]+dw4a 
    # from 5 into a
    dw5a = deltaw(l,ga,X[4])
    w5an = Wa[4]+dw5a 
    # from 6 into a
    dw6a = deltaw(l,ga,X[5])
    w6an = Wa[5]+dw6a    
    # from 7 into a
    dw7a = deltaw(l,ga,X[6])
    w7an = Wa[6]+dw7a 
    # from 8 into a
    dw8a = deltaw(l,ga,X[7])
    w8an = Wa[7]+dw8a 
    # from 9 into a
    dw9a = deltaw(l,ga,X[8])
    w9an = Wa[8]+dw9a 
    # 'a' bias
    dba = deltaw(l,ga,X[9])
    ban = Wa[9]+dba
    # update node 'a' weight
    Wa[0]=w1an
    Wa[1]=w2an
    Wa[2]=w3an
    Wa[3]=w4an
    Wa[4]=w5an
    Wa[5]=w6an
    Wa[6]=w7an
    Wa[7]=w8an
    Wa[8]=w9an
    Wa[9]=ban
    print("\nNew w1a is %8.3f, New w2a is:%8.3f, New w3a is:%8.3f, New w4a is:%8.3f"% (w1an,w2an,w3an,w4an))
    print("New w5a is %8.3f, New w6a is:%8.3f, New w7a is:%8.3f, New w8a is:%8.3f"% (w5an,w6an,w7an,w8an))
    print("New w9a is %8.3f, New bias 'a' is %8.3f\n"% (w9an,ban))

# plot error
plt.plot(errArr,label="Type 2 NFC")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend(loc='upper right')
plt.grid()
plt.show()
print("Wa",Wa)
print("W11",W11)