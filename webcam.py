import cv2
import streamlit as st
from tensorflow.keras.models import load_model
# import pydot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

st.title("FACE MASK DETETCION")

st.write("This wep app will use webcam to detect the faces in the frame and will label faces accordingly")



st.markdown('''
- Green indicates that mask is on
- Red indicates that mask is off

''')

st.write('----')



checkbox = st.checkbox("Video On")
st.caption("Check this box to ON the camera")


expander = st.expander("More Info on the algorithm working")


with expander:

    st.markdown('''## Idea:bulb::

- Once this camera is on, each frame of the webcam is fed to the face detetction model which will detetct faces and will label the faces with score or confidence.The score or confidence indication the probability with which the model was able to detect faces.

- Simutaneously these frames are fed to a Deep Convolution network which takes each frame as an input and will lable if the face has a mask on or not.

## Algoritms:computer::
- Detetcing Faces:
    - Faces in each frame of webcam is identified using Google MediaPipe library.This library included many other computer vision models such as object detection,3D - model bounding,Object Tracking etc.

- Detetcing Masks:

    - For labeling the faces if it is with mask-on or not , A Deep Convolution network is used.This network is implemented using transfer learning of MobileNetv2.This network takes a 224*224*3 size image as input is passed through various layers of convolutions and pooling to extract hierarchical features from image.

    - A set of approximateley 4000 images (with mask on and mask off) are used for traning the neural netwrok.

    - Below is the summary of the Deep Convolutional Network model.

    ''')

    st.image('model.png')



st.write("----")
# For webcam input:
video_capture = cv2.VideoCapture(0)


st.write("These values are hyperparameters for face detetcion model.Change the values using slider to see how face detetcion alogorithms works with different combination of values.")
c1,c2,c3 = st.columns((1,0.1,1))
with c1:

    model_selection = st.slider("Model Selection",0.0,1.0,0.5,step = 0.1)

    st.markdown(''' 0 or 1. 0 to select a short-range model that works best for faces within 2 meters from the camera, and 1 for a full-range model best for faces within 5 meters.''')

with c3:

    model_detect_conf = st.slider("Model Dectection Confidence",0.0,1.0,0.5,step = 0.1)

    st.markdown('''Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.''')




with st.container():

    while checkbox:
        
#         st.markdown("## **Now You are accesing the internal webcam**")
#         st.caption("Use check box to stop the video")
        frame_window = st.image([])
    
        st.write("Model loading")
        mask_detection_model = load_model('MaskDetection.h5')
        
        

        def mask(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            img = cv2.resize(img,(224,224))     # resize image to match model's expected sizing
            img = img.reshape(1,224,224,3) # return the image with shaping that TF wants

            pred = round(mask_detection_model.predict(img)[0][0],2)

            if pred < 0.5:
                pred = 100-pred*100
                out = 'Mask On'
                flag = 1
            else:
                pred = pred*100
                out = 'No Mask'
                flag = 0


            return out,str(pred),flag

        mp_Face_detect = mp.solutions.face_detection
        mp_drawings = mp.solutions.drawing_utils

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils


        with mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=model_detect_conf) as face_detection:

            
            
            
            
            while video_capture.isOpened():
                success, image = video_capture.read()

            #     if not success:
            #       print("Ignoring empty camera frame.")
            #       # If loading a video, use 'break' instead of 'continue'.
            #       continue

                # '''To improve performance, optionally mark the image as not writeable to
                # # pass by reference.'''

                #image.flags.writeable = False

                # '''Converting color from BGR to RGB'''
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # '''Inputting image to detect the face'''
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                #image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                counter = 0
                if results.detections:
                  for detection in results.detections:
                    counter = counter +1
                    x = detection.location_data.relative_bounding_box.xmin
                    y = detection.location_data.relative_bounding_box.ymin
                    w = detection.location_data.relative_bounding_box.width
                    h = detection.location_data.relative_bounding_box.height

                    out = mask(image)

                    myimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)





                    if out[-1] == 0:
                        colorbox = (0,0,255)
                        drawing = mp.python.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    else:
                        colorbox = (0,255,0)
                        drawing = mp.python.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)


                    mp_drawing.draw_detection(image, detection, bbox_drawing_spec = drawing)

                    cv2.putText(image,
                                text = f"Face {counter} : {round(detection.score[0]*100)} % ",
                                org = (int((x)*image.shape[1]),int((y)*image.shape[0])-20),
                                fontFace = cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.6,
                                color = colorbox)

                    cv2.putText(image,
                                text = " ".join(list(out[:-1])) + "%",
                                org = (int((x)*image.shape[1]),int((y)*image.shape[0])-40),
                                fontFace = cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.6,color = colorbox)


                st.write("Show Image")
                show = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                frame_window.image(show)


    else:
        video_capture.release()
        st.markdown("## **There is not available webcam**")
        st.image("Sad.png")
