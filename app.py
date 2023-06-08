import streamlit as st
from skimage.feature import hog
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm



IMG_SIZE =(32,32)


class App:

    def __init__(self):
        st.set_page_config(page_title="Tom and Jerry Classifier",initial_sidebar_state="expanded")

        # Add gradient color to background
        st.markdown(
            """
            <style>
            .css-ffhzg2 {
                background: rgb(251,66,63);
                background: radial-gradient(circle, #ff4b1f 0%, #ff9068 100%);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        title = st.markdown("<h1 style='text-align: center'>Tom and Jerry Classifier</h1>", unsafe_allow_html=True)

        __image = None
        # Create an upload button for images
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


        # If an image is uploaded, display it
        if image_file is not None:

            __image = imread(image_file)
            st.image(__image, caption='Uploaded Image')

        model_choice = st.selectbox("Choose Model", ("classification","Decision Tree", "Random Forest"))

        # Create button to trigger prediction
        if st.button("Predict"):
            # Check if image is uploaded
            if __image is not None:
                # Preprocess image
                # ------------>the work hear .........hear -->vector hog features

                __image = self.preprocess(__image)
                # Call predict function

                prediction = self.predict(__image, model_choice)
                # Display prediction
                st.write(prediction)
            else:
                st.write("Please upload an image first")

        # Set the background color to a gradient



    # should do the exact same preprocessing done in training,
    # otherwise, the model won't be guaranteed to work as expected.
    @staticmethod
    def preprocess(image):
        """
        :param image: image object (the one returned from imread)
        :return: image after processing, should be resized and transformed into a hog vector
        """
        def extract_image_features(image):
            # Resize the image to the target
            resized_image = resize(image, IMG_SIZE)

            # get hog representation of image
            observation = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=False, channel_axis=2)

            return observation

        X = np.array(extract_image_features(image))

        dataset = pd.DataFrame(X).transpose()
        return dataset[:1]

    # should use sklearn deployed model to classify image
    @staticmethod
    def predict(hog_features, model_type):
        """
        :param hog_features: hog representation of image
        :param model_type: a string representing the choosen model by the user
        :return: class, 'Tom' or 'Jerry'
        """
        if (model_type=="Random Forest"):
            RandomForest=pickle.load(open('RandomForest.pkl', 'rb'))
            y_pred = pd.DataFrame(RandomForest.predict_proba(hog_features))

        elif model_type=="Decision Tree":
            DecisionTree =pickle.load(open('DecisionTree.pkl', 'rb'))
            y_pred = pd.DataFrame(DecisionTree.predict_proba(hog_features))

        else :
            logistic = pickle.load(open('logistic.pkl', 'rb'))
            y_pred = pd.DataFrame(logistic.predict_proba(hog_features))


        return "Tom : {} \n\n  Jerry : {}".format(y_pred.iat[0,0],y_pred.iat[0,1])

app = App()
