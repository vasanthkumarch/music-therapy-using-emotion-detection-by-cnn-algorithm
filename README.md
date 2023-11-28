## MUSIC THERAPY USING EMOTION DETECTION USING REAL TIME CAPTURE USING CNN ALGORITHM
## AIM:
The primary goal of this project is to create an innovative system that seamlessly integrates real-time emotion detection using CNNs into music therapy sessions.

## FEATURES
1.Real-Time Emotion Detection

2.Data Collection Interface

3.CNN Model Architecture

4.Integration with Music Player

## REQUIREMENTS:
1.Jupyter notebook

2.Opencv

3.numpy

4.pygame

## FLOWCHART
<img width="427" alt="image" src="https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/e7976383-995d-45da-9475-c73bbd09c409">


## PACKAGES
1.install opencv
2.install numpy

## APPLICATIONS 
1.Healthcare and Mental Wellness:

Therapeutic Intervention: Implementation in hospitals, clinics, or therapy centers to assist therapists in customizing music therapy sessions based on patients' real-time emotional responses. Mental Health: Aid individuals dealing with stress, anxiety, or mood disorders by providing personalized music selections that align with their emotional states.

## PROGRAM:
### EMOTION ANALYSER:
```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load your custom image from your file system
path_to_your_image = 'C:/Users/JITHENDRA/Downloads/facial rec mini proj/images/captured_image.jpg'
custom_image = Image.open(path_to_your_image)

# Convert the image to grayscale 
custom_image = custom_image.convert('L')

# Resize the image to match the dimensions of the training images (48x48)
custom_image = custom_image.resize((48, 48))

# Convert the image to a numpy array
custom_image_array = np.array(custom_image)

# Normalize the pixel values
custom_image_array = custom_image_array.astype('float32') / 255.0

# Reshape the image to fit the model input shape
custom_image_array = custom_image_array.reshape((1, 48, 48, 1))

# Use the model to predict the emotion for this image
predicted_emotion = model.predict(custom_image_array)

# Define emotions dictionary 
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Plot the custom image and its predicted emotion
fig, ax = plt.subplots(figsize=(6, 6))

# Display the custom image
ax.imshow(custom_image_array.reshape(48, 48), cmap='gray')
ax.set_title(f'Predicted Emotion: {emotions[np.argmax(predicted_emotion)]}')
ax.axis('off')

plt.show()


# PLAY MUSIC ACCORDING TO AN EMOTION
import pygame
import os

# Initialize pygame
pygame.init()

# Path to the folder containing music files
music_folder = r"C:\Users\JITHENDRA\Downloads\facial rec mini proj\pyhton\Songs"

# Get a list of music files in the emotion folder
music_files = [file for file in os.listdir(os.path.join(music_folder, emotion)) if file.endswith(".mp3")]

# Create a dictionary to map keys to music files
music_dict = {str(i + 1): os.path.join(music_folder, emotion, file) for i, file in enumerate(music_files)}

# Play songs continuously from the specified folder
try:
    for key in music_dict:
        pygame.mixer.music.load(music_dict[key])
        pygame.mixer.music.play(-1)  # -1 makes the music play indefinitely

        while pygame.mixer.music.get_busy():
            user_input = input("Press 's' to stop the music: ")
            if user_input.lower() == 's':
                pygame.mixer.music.stop()
                break
except KeyboardInterrupt:
    pygame.mixer.music.stop()
    pygame.quit()
```
OUTPUT:
### Emotion
![image](https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/77b4f03c-03ba-4a33-b6d7-ba4ae95422a9)


### Face capturing
![image](https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/5f293e60-b57b-406d-bfb6-faa3b380878b)

### Predicted emotion
![image](https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/0a1cb02e-48b7-45ca-a78f-eddaedbed329)

### ACCURACY OF THE PREDICTED IMAGE
![image](https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/50af0cee-eb9e-4da4-87f2-9e89b527b25e)

### Emotion based song:
![image](https://github.com/vasanthkumarch/music-therapy-using-emotion-detection-by-cnn-algorithm/assets/36288975/bc940d02-668c-4611-8efe-f5862133e398)


## RESULT:
Music therapy with real-time emotion detection through CNNs signifies a revolutionary leap in personalized therapeutic interventions. By harnessing technology to interpret and adapt to individual emotional responses during music therapy sessions, it unlocks new dimensions in tailored emotional support. This innovative fusion allows for dynamic adjustments in therapeutic approaches based on detected emotions, promising a more nuanced and personalized therapeutic experience.
