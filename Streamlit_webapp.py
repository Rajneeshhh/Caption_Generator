import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Caption Generator App",
)

st.title("Welcome to Caption Generator")

BASE_DIR = ''
WORKING_DIR = ''
##############################################################################
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

#################################################################################
# load json and create model
json_file_VGG16 = open('feature_model.json', 'r')
loaded_model_json_VGG = json_file_VGG16.read()
json_file.close()
feature_model = model_from_json(loaded_model_json_VGG)
# load weights into new model
feature_model.load_weights("feature_model.h5")
#################################################################################

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)
################################################################################
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
############################################################################################
# preprocess the text
clean(mapping)
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)
# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length
#################################################################################
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
#################################################################################
# generate caption for an image
def predict_caption(model, image1, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image1, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text
#################################################################################

def generate_caption(image_name):
    # convert image pixels to numpy array
    image_name = img_to_array(image_name)
    # reshape data for model
    image_name = image_name.reshape((1, image_name.shape[0], image_name.shape[1], image_name.shape[2]))
    # preprocess image for vgg
    image_name = preprocess_input(image_name)
    # extract features
    feature = feature_model.predict(image_name, verbose=0)
    #########################################################
    # Load image
    # image = Image.open(uploaded_file)
    # predict the caption
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    st.markdown('--------------------Caption For Uploaded Image--------------------')
    st.markdown(y_pred)
#################################################################################
st.title("Image Uploader")
    
# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    newsize = (224, 224)
    image = image.resize(newsize)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    generate_caption(image)
