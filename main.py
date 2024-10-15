import streamlit as st
import tensorflow as tf
import numpy as np

# Model Prediction from Tensorflow
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#TheSidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("SYSTEM FOR PLANT DISEASE RECOGNITION")
    image_path = "nana.png.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the System for Plant Disease Recognition! 🌿🔍
    
    Our mission is to provide an easy way of identifying diseases in plants. 
    By uploading an image of a plant, our system can analyze it for disease symptoms. Let us in unison take on the quest of protecting our crops and reaping better harvests!

    ### How It Works
1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of the plant suspected to be suffering from diseases.
2. **Analysis:** Our system will process the uploaded image with advanced algorithms to spot potential diseases.
3. **Results:** Proceed to view the results and recommendations for next steps.

### Why Choose Us?
Accuracy: ours is a system designed using the latest machine learning methods to ensure highly accurate disease diagnosis.
- User-Friendly: Easy and intuitive interface for seamless user experience.
- Quick and Efficient: Get your results back in a matter of seconds for fast decision-making.

### Getting Started
Click the **Disease Recognition** page on the sidebar and upload an image to see the magic of our Plant Disease Recognition System!

### About Us
Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    image_path = "nana.png.png"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 87,000 RGB images taken from healthy and diseased crop leaves representing 38 classes of diseases. The entire dataset is split according to the 80/20 ratio of the training and validation sets, respectively, but the directory structure should be preserved.
                Afterwards, a new folder with 33 test images is created for the purpose of prediction.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    image_path = "pepper.jpg"
    st.image(image_path,use_column_width=True)
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))


