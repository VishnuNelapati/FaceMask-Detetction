# FaceMask-Detetction
Detecting the faces in the webcam feed and labelling the faces if the mask is on or not.

Idea💡:
Once this camera is on, each frame of the webcam is fed to the face detetction model which will detetct faces and will label the faces with score or confidence.The score or confidence indication the probability with which the model was able to detect faces.

Simutaneously these frames are fed to a Deep Convolution network which takes each frame as an input and will lable if the face has a mask on or not.

Algoritms💻:
Detetcing Faces:

Faces in each frame of webcam is identified using Google MediaPipe library.This library included many other computer vision models such as object detection,3D - model bounding,Object Tracking etc.
Detetcing Masks:

For labeling the faces if it is with mask-on or not , A Deep Convolution network is used.This network is implemented using transfer learning of MobileNetv2.This network takes a 2242243 size image as input is passed through various layers of convolutions and pooling to extract hierarchical features from image.

A set of approximateley 4000 images (with mask on and mask off) are used for traning the neural netwrok.


