# # import streamlit as st
# # import numpy as n
# # from PIL import Image
# # import cv2

# # st.title("Hello GeeksForGeeks !!!")

# import streamlit as st
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Helper function to simulate pose keypoints
# def detect_pose(image):
#     """
#     Simulated function to detect pose keypoints in an image.
#     For simplicity, this returns mock data representing keypoints.
#     """
#     # Example keypoints (x, y) as fractions of image width and height
#     keypoints = [
#         (0.5, 0.2),  # Nose
#         (0.45, 0.3), # Left Shoulder
#         (0.55, 0.3), # Right Shoulder
#         (0.4, 0.5),  # Left Elbow
#         (0.6, 0.5),  # Right Elbow
#         (0.35, 0.7), # Left Wrist
#         (0.65, 0.7), # Right Wrist
#         (0.5, 0.8),  # Hip
#         (0.45, 1.0), # Left Knee
#         (0.55, 1.0), # Right Knee
#     ]
#     return keypoints

# # Helper function to draw pose on an image using Matplotlib
# def plot_pose(image, keypoints):
#     """
#     Plot pose keypoints on an image using Matplotlib.
#     """
#     fig, ax = plt.subplots()
#     ax.imshow(image)
    
#     # Scatter plot for keypoints
#     x_coords = [kp[0] * image.shape[1] for kp in keypoints]
#     y_coords = [kp[1] * image.shape[0] for kp in keypoints]
#     ax.scatter(x_coords, y_coords, c="red", label="Keypoints")

#     # Connect keypoints to form skeleton
#     skeleton = [
#         (0, 1), (0, 2), # Nose to shoulders
#         (1, 3), (3, 5), # Left arm
#         (2, 4), (4, 6), # Right arm
#         (1, 7), (2, 7), # Shoulders to hip
#         (7, 8), (7, 9), # Hip to knees
#     ]
#     for joint in skeleton:
#         x = [x_coords[joint[0]], x_coords[joint[1]]]
#         y = [y_coords[joint[0]], y_coords[joint[1]]]
#         ax.plot(x, y, c="blue")

#     ax.axis("off")
#     st.pyplot(fig)

# # Streamlit app
# st.title("Human Pose Estimation App (Matplotlib)")
# st.write("Upload an image, and the app will estimate the human pose using Matplotlib.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     # Read and display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Convert to numpy array (OpenCV format)
#     image_np = np.array(image)

#     # Simulate pose detection
#     keypoints = detect_pose(image_np)

#     # Plot pose using Matplotlib
#     plot_pose(image_np, keypoints)

import cv2
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image

BODY_PARTS={ "Nose":0,"Neck":1,"RShoulder":2,"Relbow":3,"Rwrist":4,"LShoulder":5,"Lelbow":6,"Lwrist":7,"RHip":8,"RKnee":9,"Rankle":10,"LHip":11,"LKnee":12,"Lankle":13,"REye":14,"LEye":15,"Rear":16,"Lear":17,"Background":18


}
POSE_PARTS=[   ["Neck","RShoulder"],["Neck","LShoulder"],["RShoulder","Relbow"],["Relbow","Rwrist"],["LShoulder","Lelbow"],["Lelbow","Lwrist"],["Neck","RHip"],["RHip","RKnee"],["RKnee","Rankle"],["Neck","LHip"],["LHip","LKnee"],["LKnee","Lankle"],["Neck","Nose"],["Nose","REye"],["REye","Rear"],["Nose","LEye"],["LEye","Lear"]

]







width=350
height=350


inwidth=width
inheight=height


net= cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.title("Human Pose Estimation App (Matplotlib)")
st.write("Upload an image, and the app will estimate the human pose using Matplotlib.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array (OpenCV format)
    image_np = np.array(image)

    # Simulate pose detection
    keypoints = detect_pose(image_np)

    # Plot pose using Matplotlib
    plot_pose(image_np, keypoints)



thres=0.2
def posedetector(frame):
    frameWidth=frame.shape[1]
    frame.Height=frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame,1.0,(inwidth,inheight),(127.5,127.5,127.5),swapRB=True,crop=false))

    out=net.forward()
    out=out[:,:19,:,:]

    assert(len(BODY_PARTS))==out.shape([1])

    points=[]

    for i in range(len(BODY_PARTS)):

        heatmap=out[0,i,:,:]
        _,conf,_point=cv2.minMaxLoc(heatmap)

        x=(frameWidth*point [0])/out.shape[3]
        y=(framewidth * point[1])/out.shape[2]

        points.append((int(x),int(y)) if conf > thres else None)


        for pair in POSE_PAIRS:
            partfrom=pair[0]
            partto=pair[1]
            assert(partfrom in BODY_PARTS)
            assert(partto in BODY_PARTS)

            idfrom=BODY_PARTS[partfrom]
            idTo=BODY_PARTS[partTo]

            if points[idfrom] and points[idTo]:
                cv2.line(frame,points[idfrom],points[idTo],(0,255,0),3)

                cv2.ellipse(frame,points[idfrom],(3,3),0,0,360,(0,0,255),cv2.FILLED)
                cv2.ellipse(frame,points[idTo],(3,3),0,0,360,(0,0,255),cv2.FILLED)
        
        t,_=net.getPerfProfile()

        return frame

   input =cv2.imread("image.jpg")


  output= posedetector(input)