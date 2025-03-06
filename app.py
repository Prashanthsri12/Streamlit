import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary classification web app")
    st.sidebar.title("Binary classification web app")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data(persist=True)
    def load_data():
        data_path = "C:/Users/Prashanth/Documents/Workspace/mushrooms.csv"
        data = pd.read_csv(data_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df['type'].values
        X = df.drop("type", axis=1).values
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
        return X_train,X_test,y_train,y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)


    class_names = ['edible','poisonous']
    df = load_data()
    X_train,X_test,y_train,y_test = split(df)

    classifier = st.sidebar.selectbox("Choose Classifier",("Support Vector Machine (SVM)","Logistic Regression","Random Forest"))
    
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01,key='C')
        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale","auto"),key='gamma')
        metrics_list = st.sidebar.multiselect("Metrics to plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            with st.spinner(text="In progress"):
                bar = st.progress(0)
                model = SVC(C=C,kernel=kernel,gamma=gamma)
                bar.progress(10)
                model.fit(X_train,y_train)
                bar.progress(50)
                y_pred = np.array(model.predict(X_test))
                accuracy = round(model.score(X_test,y_test),2)
                precision = round(precision_score(y_test,y_pred,labels=class_names),2)
                recall = round(recall_score(y_test,y_pred,labels=class_names),2)
                bar.progress(70)
                st.write("Accuracy: ",accuracy)
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                bar.progress(90)
                plot_metrics(metrics_list)
                bar.progress(100)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='n_estimators')
        max_depth = st.sidebar.number_input("Max depth of the tree",1,20,step=1,value=10,key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True","False"), key='bootstrap')
        metrics_list = st.sidebar.multiselect("Metrics to plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            with st.spinner(text="In progress"):
                bar = st.progress(0)
                if bootstrap == 'True':
                    bs = True
                else:
                    bs = False
                model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bs, n_jobs=-1)
                bar.progress(10)
                model.fit(X_train,y_train)
                bar.progress(50)
                y_pred = np.array(model.predict(X_test))
                accuracy = round(model.score(X_test,y_test),2)
                precision = round(precision_score(y_test,y_pred,labels=class_names),2)
                recall = round(recall_score(y_test,y_pred,labels=class_names),2)
                bar.progress(70)
                st.write("Accuracy: ",accuracy)
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                bar.progress(90)
                plot_metrics(metrics_list)
                bar.progress(100)
            
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01,key='C_LR')
        max_iter = st.sidebar.slider("Max Iterations",100,1000,key = 'max_iter')
        metrics_list = st.sidebar.multiselect("Metrics to plot",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            with st.spinner(text="In progress"):
                bar = st.progress(0)
                model = LogisticRegression(C=C, max_iter=max_iter )
                bar.progress(10)
                model.fit(X_train,y_train)
                bar.progress(50)
                y_pred = np.array(model.predict(X_test))
                accuracy = round(model.score(X_test,y_test),2)
                precision = round(precision_score(y_test,y_pred,labels=class_names),2)
                recall = round(recall_score(y_test,y_pred,labels=class_names),2)
                bar.progress(70)
                st.write("Accuracy: ",accuracy)
                st.write("Precision: ", precision)
                st.write("Recall: ", recall)
                bar.progress(90)
                plot_metrics(metrics_list)
                bar.progress(100)

    
    

    st.sidebar.write("___")
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset (Classification)")
        st.write(df)
        
    
    

if __name__ == '__main__':
    main()


