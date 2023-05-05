from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import cv2

"""
    Important!

    If testing this code, please change 'TRAINED_MODEL_PATH' to where project files have been saved
"""

# Define paths
CONFIG_PATH = "C:/Users/VVL/Desktop/LenkevicsValdisCreativePiece/Final_Year_Project_Models/mmdetection/mmdetect_custom_model_config.py"
MODEL_PATH = "C:/Users/VVL/Desktop/LenkevicsValdisCreativePiece/Final_Year_Project_Models/mmdetection/epoch_3.pth"

# Set device
#DEVICE = 'cuda:0'
DEVICE = "cpu"

# Set confidence thr
CONFIDENCE_LEVEL = 0.5

# Load config file
config_file = Config.fromfile(CONFIG_PATH)

# Load model
model = init_detector(config_file, MODEL_PATH, device=DEVICE)

# Launch live camera to detect results
camera = cv2.VideoCapture(0)

print('Press "Esc", "q" or "Q" to exit.')
while True:
    ret_val, img = camera.read()
    result = inference_detector(model, img)

    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord('q') or ch == ord('Q'):
        break

    model.show_result(
        img, result, score_thr=CONFIDENCE_LEVEL, wait_time=1, show=True)