# A cheatsheet: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py
# Finish later. ;)

import streamlit as st
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
MODEL_PATH = ""





# TODO: Make this streaming.
def mk_images(video_pat):
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        yield img


# Resize and transform an image into a Pytorch
# Resize to 512 x 512. Is this the best size?
def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)









def image_loader(image):
    """ Transform an image into a Tensor.
    """
    loader = get_valid_transforms()
    tensor = loader(image=image)["image"].float()
    tensor = tensor.unsqueeze(0)
    # TODO: Should be passed to CPU maybe?
    return tensor  #assumes that you're using GPU


# DS code from here: https://www.kaggle.com/artkulak/2class-object-detection-inference-with-filtering
# Idea: take a video and zoom onn impact if any.

# TODO: Adapt this so that we have one prediction at a time. 
def make_predictions(images, score_threshold=0.5):
    images = torch.stack(images).cuda().float()
    box_list = []
    score_list = []
    with torch.no_grad():
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]   
            label = det[i].detach().cpu().numpy()[:,5]
            # useing only label = 2
            indexes = np.where((scores > score_threshold) & (label == 2))[0]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            box_list.append(boxes[indexes])
            score_list.append(scores[indexes])
    return box_list, score_list



# Run an object detection on the selected video.
# TODO: Add a spinner while the video is being processed?

ACTIONS = ["play", "predict"]

def play_video():
    uploaded_file = st.file_uploader("Select a video to play: ")
    if uploaded_file is not None:
         bytes_data = uploaded_file.read()
 
         st.video(bytes_data)

def predict_image():
    uploaded_file = st.file_uploader("Select an image to predict: ")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=512)
        # To numpy array
        image = np.array(image)

        tensor = image_loader(image)
        mean_tensor = tensor.mean()
        st.text("The mean of the loaded image is: ")
        st.write(mean_tensor)


selection = st.sidebar.selectbox("What do you want to do? ", ACTIONS)
if selection == "play":
    play_video()
elif selection == "predict":
    predict_image()