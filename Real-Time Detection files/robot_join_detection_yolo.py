from ultralytics import YOLO

"""
    Important!

    If testing this code, please change 'TRAINED_MODEL_PATH' to where project files have been saved
"""

# Decalre model .pt file path
TRAINED_MODEL_PATH = 'C:/Users/VVL/Desktop/Final_Year_Project_Practical_19074904/Final_Year_Project_Models/yolo/'

# Load a model
model = YOLO(TRAINED_MODEL_PATH+"best.pt")  # load custom model trained on Colab

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
SOURCE = "0" # Webcam

# Display camera
DISPLAY_WEBCAM = True

# Set confidence thr
CONFIDENCE_LEVEL = 0.5

# Predict using Webcam, Source="0" == webcam, show, displays the camera and detections
results = model.predict(source=SOURCE, show=DISPLAY_WEBCAM, conf=CONFIDENCE_LEVEL)



