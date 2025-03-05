import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
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
        data_path = "C:/Users/Prashanth/Documents/Workspace/Streamlit/ML web app/home/coder/coursera/Project/mushrooms.csv"
        data = pd.read_csv(data_path)
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset (Classification)")
        st.markdown("*Encoded data")
        st.write(df)
        
    st.number_input("Pick a number", 0, 10)

if __name__ == '__main__':
    main()


