import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model


# Settings:
## Extra CSS:
st.set_page_config(page_title='Real Estate Investment Predictor', page_icon=':bar_chart:', layout='wide')
hide_st_style = '''
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_st_style, unsafe_allow_html=True)
cp_priority_dict = {
    'Category I Exclusively Medium-high Residential Zone': 1,
    'Commercial Zone': 2,
    'Quasi-industrial Zone': 3,
    'Category I Residential Zone': 4,
    'Category I Exclusively Low-story Residential Zone': 5,
    'Quasi-residential Zone': 6,
    'Category II Residential Zone': 7,
    'Non-divided City Planning Area': 8,
    'Neighborhood Commercial Zone': 9,
    'Industrial Zone': 10,
    'Outside City Planning Area': 11,
    'Category II Exclusively Low-story Residential Zone': 12,
    'Urbanization Control Area': 13,
    'Category II Exclusively Medium-high Residential Zone': 14,
    'Exclusively Industrial Zone': 15,
    'Quasi-city Planning Area': 16
}


pre_labelling = {
    'Tokyo': 1,
    'Aichi': 2,
    'Shiga': 3,
    'Shizuoka': 4,
    'Fukui': 5,
    'Toyama': 6,
    'Osaka': 7,
    'Nagano': 8,
    'Ishikawa': 9,
    'Hiroshima': 10,
    'Kyoto': 11,
    'Mie': 12,
    'Kanagawa': 13,
    'Yamanashi': 14,
    'Okinawa': 15,
    'Kagawa': 16,
    'Gunma': 17,
    'Tochigi': 18,
    'Niigata': 19,
    'Hyogo': 20,
    'Oita': 21,
    'Gifu': 22,
    'Hokkaido': 23,
    'Ibaraki': 24,
    'Yamaguchi': 25,
    'Miyagi': 26,
    'Fukushima': 27,
    'Tottori': 28,
    'Shimane': 29,
    'Fukuoka': 30,
    'Kumamoto': 31,
    'Ehime': 32,
    'Chiba': 33,
    'Yamagata': 34,
    'Saga': 35,
    'Iwate': 36,
    'Tokushima': 37,
    'Nara': 38,
    'Saitama': 39,
    'Kochi': 40,
    'Wakayama': 41,
    'Miyazaki': 42,
    'Akita': 43,
    'Nagasaki': 44,
    'Kagoshima': 45,
    'Okayama': 46,
    'Aomori': 47
}


# Loading Resources:
@st.cache_resource
def get_model2():
    model = load_model('mixed_v1.h5')
    return model

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    resized = im.resize((128, 128))
    image = np.array(resized)
    return image


model2 = get_model2()

uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

if uploadFile is not None:
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
    
    cp = st.multiselect(
    'Select The Appropriate City Type:',
    options= ['Category I Exclusively Medium-high Residential Zone',
    'Commercial Zone',
    'Quasi-industrial Zone',
    'Category I Residential Zone',
    'Category I Exclusively Low-story Residential Zone',
    'Quasi-residential Zone',
    'Category II Residential Zone',
    'Non-divided City Planning Area',
    'Neighborhood Commercial Zone',
    'Industrial Zone',
    'Outside City Planning Area',
    'Category II Exclusively Low-story Residential Zone',
    'Urbanization Control Area',
    'Category II Exclusively Medium-high Residential Zone',
    'Exclusively Industrial Zone',
    'Quasi-city Planning Area'],
    default='Industrial Zone',
    max_selections=1)
    if cp:
        city_planning = cp_priority_dict[cp[0]]
    else:
        city_planning = 0
    # print(img)
    citi = st.multiselect(
        'Select The Querry Prefecture:',
        options=['Fukui', 'Nagano', 'Nagasaki', 'Okayama', 'Fukushima', 'Mie',
        'Saitama', 'Wakayama', 'Ishikawa', 'Tokushima', 'Toyama', 'Ehime',
        'Tottori', 'Yamanashi', 'Yamaguchi', 'Saga', 'Miyazaki', 'Kyoto',
        'Hyogo', 'Tokyo', 'Kumamoto', 'Aichi', 'Kanagawa', 'Shizuoka',
        'Fukuoka', 'Oita', 'Yamagata', 'Akita', 'Chiba', 'Kochi',
        'Tochigi', 'Miyagi', 'Hokkaido', 'Okinawa', 'Iwate', 'Niigata',
        'Kagawa', 'Gunma', 'Ibaraki', 'Shiga', 'Nara', 'Gifu', 'Shimane',
        'Hiroshima', 'Osaka', 'Aomori', 'Kagoshima'],
        default='Okayama',
        max_selections=1
    )
    bed = st.number_input('No. of bedrooms:', step=1, value=2, min_value=0)
    bath = st.number_input('No. of bathrooms:', step=1, value=2, min_value=0)
    sqft = st.number_input('Sqft:', step=1, value=700, min_value=0)

    pred_holder = pd.DataFrame(columns=['image_id', 'citi', 'bed', 'bath', 'sqft'])
    pred_holder.loc[len(pred_holder.index)] = [0, pre_labelling[citi[0]], bed, bath, sqft]

    img = np.expand_dims(img, axis=0)
    print(pred_holder.values.shape)
    print(img.shape)

    tf_pred_holder = tf.constant(pred_holder, dtype=tf.float64)
    tf_img_holder = tf.constant(img)
    pred_val = model2.predict([tf_pred_holder, tf_img_holder], verbose=0)[:][0][0]
    pres = model2.predict([pred_holder, img])

    plot_predict = st.button('Predict')
    if plot_predict:
        st.subheader(f'Predicted Plot Price: {pred_val*145:.2f} ¥')

else:
    st.write("Make sure you image is in JPG/PNG Format.")
