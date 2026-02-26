import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Salary Pridiction")

df = pd.read_csv("salary_prid.csv")
st.dataframe(df)


model = LinearRegression()
model.fit(df[["exp"]],df["Salary"])


experience = st.number_input("No of exp: ",1,4)

pred = model.predict([[experience]])
st.write(pred)
