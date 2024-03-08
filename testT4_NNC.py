from NNfunction import *
import pandas as pd
import matplotlib.pyplot as plt

raw_df=pd.read_csv("testdata.data") # read data from .data
filtered_df=raw_df[raw_df['at10'] == 4]
allData_df=filtered_df.dropna()  # drop rows with NaN values
allData=allData_df.values.astype(float) # set values type
Wa=([0.11375834859627197, 0.10839472965210996, 0.10938519147216079, 0.10710828285649675, 0.10850767681504449, 0.11201449404294299, 0.10896577908820344, 0.10863932950619977, 0.10392054353137038, 1.0021729049390693]) #[w1a,w2a,w3a,w4a,w5a,w6a,w7a,w8a,w9a,Bias]
W11=([1.0658069795163223, 1.970160365622272]) # [wa11,Bias]
l=-0.9 # learning rate
count=1
errArr=[]
d11=1
epochArr=[]

for i in range(len(allData)):
    curr=allData[i]
    X=list(curr[1:10])+[1.0] # input values

    print("\nEpoch:", count)
    epochArr.append(count)
    print("\n--------------------------------------------------")

    # node a
    va=Nout(X,Wa)
    ya=sigmoid(va)
    print("\nSum(V) of node 'a' is: %8.3f, Y from node 'a' is: %8.3f" % (va,ya))

    # node 11
    X11=[ya,1]
    v11=Nout(X11,W11)
    y11=sigmoid(v11)
    print("\nSum(V) of node 11 is: %8.3f, Y from node 11 is: %8.3f" % (v11,y11))

    # cal error
    err=d11-y11
    errArr.append(err)
    print(err)

    print("\n--------------------------------------------------")
    count = count + 1

# plot test result
plt.axis([0, 100, 0, 1])  
plt.scatter(epochArr,errArr,label="Type 4 Test")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend(loc='upper right')
plt.grid()
plt.show()
print(sum(errArr)/len(errArr))


