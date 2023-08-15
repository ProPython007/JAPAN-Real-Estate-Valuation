import streamlit as st

import numpy as np
import pandas as pd

# import sqlite3
# conn = sqlite3.connect('student_feedback.db')
# c = conn.cursor()
    

# def create_table():
#     c.execute('CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 INTEGER, Q3 INTEGER)')

# def add_feedback(date_submitted, Q1, Q2, Q3):
#     c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3) VALUES (?,?,?)',(date_submitted,Q1, Q2, Q3))
#     conn.commit()

def main():

    st.title("Investers Feedback")

    d = st.date_input("Today's date",None, None, None, None)
    
    
    
    question_1 = st.slider('rate our investement details?', 0,10)
    st.write('You selected:', question_1) 
    
    

    question_2 = st.selectbox('Are u happy with our product then select ?',('Not That much good','Happy', 'Wonderful','Can be improve more'))
    st.write('You selected:', question_2)

   

    question_3 = st.text_input('What could have been better?', max_chars=50)

    if st.button("Submit feedback"):
        # create_table()
        # add_feedback(d, question_1, question_2, question_3)
        st.success("Feedback submitted")

if __name__ == '__main__':
    main()