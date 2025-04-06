import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

st.markdown("""
<style>
/* General App Styling */
.stApp {
    background-color: #0f0f3e;  /* Deep Blue */
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
    padding: 10px;
}

/* Hide Streamlit Footer and Menu */
#MainMenu, footer {
    visibility: hidden;
}

/* Headings */
h1, h2, h3, h4 {
    color: #ffffff;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #1b1b52;
    color: white;
}

/* File Uploader and Buttons */
button[kind="primary"] {
    background-color: #4A90E2;
    border-radius: 12px;
    color: white;
    padding: 0.5rem 1rem;
}

button[kind="primary"]:hover {
    background-color: #357ABD;
}

/* Markdown in Cards */
.block-container {
    padding: 1.5rem;
}

/* Cards or Containers */
.stMarkdown {
    background-color: #1e1e4e;
    padding: 1.2rem;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
}

/* Audio Player */
audio {
    border-radius: 10px;
    margin-top: 10px;
}

/* Balloons Confetti */
.css-1ekf893 {
    background-color: #0f0f3e !important;
}
</style>
""", unsafe_allow_html=True)


#Function
@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("Trained_model.h5")
  return model


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]



#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    image_path = "music_genre_home.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
""")



#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)

    

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3 is not None:
        # Save uploaded file
        import os
        os.makedirs("Test_Music", exist_ok=True)
        filepath = os.path.join("Test_Music", test_mp3.name)
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())
        st.success(f"Uploaded file saved as {test_mp3.name}")

        # Play audio
        if st.button("Play Audio"):
            st.audio(test_mp3)

        # Predict
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
                st.markdown(f"**:blue[Model Prediction:] It's a  :red[{label[result_index]}] music**")
