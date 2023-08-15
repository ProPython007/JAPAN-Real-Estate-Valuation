# app.py, run with 'streamlit run app.py'
import pandas as pd
import streamlit as st


df = pd.read_csv("file.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
rename = {'amt':'Trade Price (Yen)', 'lat':'Latitude coordinate', 'long':'Londitude Coordinate'}
df.rename(columns = rename, inplace = True)
df_sorted = df.sort_values(by = 'Trade Price (Yen)', ascending= False)
df_sorted.drop('col', axis=1, inplace=True)
st.title("Best Investments according to your Preference!")  # add a title
# print(df_sorted)

custom_css = """
<style>
.table-container {
    background-color: red;
    font-family: Arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

.table-container th, .table-container td {
    padding: 8px;
    text-align: center;
    border-bottom: 1px solid #ddd;
}

.table-container th {
    background-color: #f2f2f2;
}

.table-container tr:hover {
    background-color: #f5f5f5;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
df_sorted.reset_index(inplace=True)
df_sorted.drop('index', axis=1, inplace=True)
st.table(df_sorted)