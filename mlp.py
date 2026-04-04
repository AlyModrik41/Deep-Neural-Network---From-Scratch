import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(v):
    return 1 / (1 + np.exp(-v))
def tanh(v):
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
            curr_out = tanh(v)
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

def confusion_matrix(y_true, y_pred, classes):
    matrix = [[0] * classes for i in range(classes)]
    for y_values in range(len(y_pred)):
        pred = y_pred[y_values]
        true = y_true[y_values]
        matrix[true][pred] += 1
    return matrix

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return (correct / len(y_true)) * 100



df=pd.read_csv('df_encoded.csv')
# print(df.head())
X=df.drop(columns=['Species'])
y=df['Species']
class_names = ['Adelie', 'Chinstrap', 'Gentoo']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42,stratify=y)

# Users Input List
st.markdown("""
    <style>
    /* main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* text color */
    .stApp, .stMarkdown, label, .stSelectbox, .stTextInput {
        color: #ffffff !important;
    }
    
    /* input fields */
    .stTextInput input, .stNumberInput input {
        background-color: #16213e;
        color: #ffffff;
        border: 1px solid #4a90d9;
        border-radius: 8px;
    }
    
    .stTextInput input:hover, .stNumberInput input:hover {
        border: 1px solid #ffffff;
        background-color: #0f3460;
    }
    .stSelectbox div:hover {
        border-color: #ffffff;
        background-color: #0f3460;
    }
            
    /* button */
    .stButton button {
        background-color: #4a90d9;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton button:hover {
        background-color: #357abd;
    }
    .stButton button:hover {
        background-color: #0f3460 !important;
        color: #ffffff !important;
        border: 1px solid #4a90d9;
    }
    .stCheckbox:hover {
        color: #4a90d9 !important;
    }
    /* title */
    h1 {
        color: #4a90d9 !important;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🐧 penguins Neural Network Classifier")

n_layers   = st.number_input("Number of Hidden Layers", min_value=1, value=2, step=1)
neurons    = st.text_input("Neurons per Layer (comma separated)", "6,4")
eta        = st.number_input("Learning Rate (η)", value=0.01, format="%.4f")
epochs     = st.number_input("Epochs", min_value=1, value=1000, step=1)
b_use      = st.checkbox("Add Bias", value=True)
activation = st.selectbox("Activation Function", ["sigmoid", "tanh"])

if st.button("Train Network"):
    neurons_list = [int(n.strip()) for n in neurons.split(",")]
    architecture_list=[X.shape[1]]+neurons_list+[3]

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
    predict     = list(predict)
    true_labels = list(true_labels)

    # ── RESULTS ───────────────────────────────────────
    train_predict = []
    one_hot_train = pd.get_dummies(y_train).values
    for i in range(len(X_train)):
        sample    = X_train.iloc[i].values.astype(float).reshape(-1, 1)
        acc_array = forward_step(sample, weights, bias, activation, b_use)
        train_predict.append(np.argmax(acc_array[-1]))

    true_train = list(np.argmax(one_hot_train, axis=1))

    train_acc = accuracy(true_train, train_predict)
    test_acc  = accuracy(true_labels, predict)

   
    cm    = confusion_matrix(true_labels, predict, len(class_names))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    

    st.write(f"### 🎯 Train Accuracy: {train_acc:.2f}%")
    st.progress(int(train_acc))
    st.write(f"### 🧪 Test Accuracy: {test_acc:.2f}%")
    st.progress(int(test_acc))

    st.write("### 📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={"size": 20}, linewidths=2, linecolor='white')
    ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
    ax.set_ylabel("Actual", fontsize=14, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=16, pad=15)
    ax.tick_params(axis='both', labelsize=13)
    fig.patch.set_facecolor('#746aff')
    ax.set_facecolor("#746aff")
    st.pyplot(fig)



            

