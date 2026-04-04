import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

def sigmoid(v):
    return 1 / (1 + np.exp(-v))
def tabh(v):
    return np.tanh(v)
def forward_step(sample , weights ,  bias , actevation_name , b_use ):
    acc_array = [sample]
    curr_smaple =sample
    for i in range (len(weights)):
        w = weights[i]
        v = np.dot(w , curr_smaple)
        if b_use:
            b = bias[i]
            v+=b
        if actevation_name == 'sigmoid':
            curr_out = sigmoid(v)
        elif actevation_name == 'tanh':
            curr_out = tabh(v)
        acc_array.append(curr_out)
        curr_smaple = curr_out
    return acc_array

def compute_errors(weights,acc_array,one_hot_vector,activation_name):
    error=[]
    output=acc_array[-1]
    if activation_name=='sigmoid':
        derivative=output*(1-output)
    elif activation_name=='tanh':
        derivative=1-output*output
    error.append((one_hot_vector-output)*derivative)
    error_inst=error[-1]
    for i in range(1,len(weights)):
        output=acc_array[-i-1]
        if activation_name=='sigmoid':
            derivative=output*(1-output)
            error_inst=np.dot(weights[-i].T,error_inst)*(derivative)
        elif activation_name=='tanh':
            derivative=1-output*output
            error_inst=np.dot(weights[-i].T,error_inst)*(derivative)
        error.append(error_inst)
    
    return error

def update_weights(weights,acc_array,errors,eta,bias,b_use):
    for i in range(1,len(weights)+1):
        weights[-i]=weights[-i]+eta*np.dot(errors[i-1],acc_array[-(i+1)].T)
        if b_use==True:
            bias[-i]=bias[-i]+eta*errors[i-1]

    return weights,bias





df=pd.read_csv('df_encoded.csv')
# print(df.head())
X=df.drop(columns=['Species'])
y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# Users Input List
eta = float(input("Enter learning rate: "))
epochs = int(input("Enter epochs: "))
n_hidden = int(input("Enter number of hidden layers: "))
neurons = [int(input(f"Neurons in hidden layer {i+1}: ")) for i in range(n_hidden)]
b_use = input("Use bias? (yes/no): ") == 'yes'
activation = input("Activation (sigmoid/tanh): ")

architecture_list=[X.shape[1]]+neurons+[3]

one_hot_vector=pd.get_dummies(y_train).values
weights=[]
bias=[]

for i in range(1,len(architecture_list)):
    weights.append(np.random.randn(int(architecture_list[i]),int(architecture_list[i-1]))*0.01)
    bias.append(np.zeros((int(architecture_list[i]),1)))


for j in range(0,epochs):
    for i in range(len(X_train)):
        sample=X_train.iloc[i].values.reshape(-1,1)
        sample_one_hot=one_hot_vector[i].reshape(-1,1)
        acc_array=forward_step(sample,weights,bias,activation,b_use)
        errors=compute_errors(weights,acc_array,sample_one_hot,activation)
        weights,bias=update_weights(weights,acc_array,errors,eta,bias,b_use)

one_hot_vector_test=pd.get_dummies(y_test).values
predict=[]
for i in range(len(X_test)):
        sample=X_test.iloc[i].values.reshape(-1,1)
        sample_one_hot=one_hot_vector_test[i].reshape(-1,1)
        acc_array=forward_step(sample,weights,bias,activation,b_use)
        predict.append(np.argmax(acc_array[-1]))
        # print(np.argmax(acc_array[-1]))

true_labels=np.argmax(one_hot_vector_test,axis=1)
print(confusion_matrix(true_labels,predict))
print(accuracy_score(true_labels,predict))


        

