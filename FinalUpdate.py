import cv2 as cv2
import numpy as np
import mediapipe as mp
import math
from sklearn.metrics import mean_squared_error
#Load mediapipe face mesh module
mp_face_mesh = mp.solutions.face_mesh

#Define landmarks indices for eyes and iris
LEFT_EYE=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS=[474, 475, 476, 477]
LEFT_IRIS=[469, 470, 471, 472]
L_H_LEFT=[33]
L_H_RIGHT=[133]
R_H_LEFT=[362]
R_H_RIGHT=[263]

#Function to calculate euclidean distance between two points
def euclidean_distance(point1,point2):
    x1,y1 =point1.ravel()
    x2,y2 =point2.ravel()
    distance=math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

#Function to detemine iris position based on landmarks 
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position=""
    if ratio<=0.42:
        iris_position ="right"
    elif ratio>0.42 and ratio<=0.57:
        iris_position="center"
    else :  
        iris_position="left"
    return iris_position, ratio          

# Predefined ground truth values for left and right iris positions
# You need to replace these with actual ground truth values
# For demonstration, let's assume ground truth values are (100, 100) for left iris and (200, 200) for right iris
left_iris_ground_truth = np.array([100, 100])
right_iris_ground_truth = np.array([200, 200])

#Open video capture
cap=cv2.VideoCapture(0)

# Initialize lists to store predicted iris positions
left_iris_positions = []
right_iris_positions = []

#Intialize face mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as face_mesh:
    
    while True:
        #Read frame from video capture
        ret,frame= cap.read()
        if not ret:
            break

        #Flip frame horizontally
        frame =cv2.flip(frame,1)
        rgb_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_h,img_w=frame.shape[:2]
        #Process frame using face mesh model
        results= face_mesh.process(rgb_frame)      
        if results.multi_face_landmarks:
            #print(results.multi_face_landmarks[0].landmark)
            #Extract mesh points 
            mesh_points=np.array([np.multiply([p.x ,p.y], [img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            #Calculate and draw polygons around left and right eyes
            left_eye_polygon = np.array(mesh_points[LEFT_EYE], np.int32)
            right_eye_polygon = np.array(mesh_points[RIGHT_EYE], np.int32)
            cv2.polylines(frame, [left_eye_polygon], True, (255, 0, 0), 1)
            cv2.polylines(frame, [right_eye_polygon], True, (0, 255, 0), 1)

         
            #print(mesh_points.shape)
            #cv2.polylines(frame,[mesh_points[LEFT_IRIS]],True,(255,0,0),1, cv2.LINE_AA)
            #cv2.polylines(frame,[mesh_points[RIGHT_IRIS]],True,(0,255,0),1, cv2.LINE_AA)

            #Calculate and draw circles around left and right iris  
            (l_cx,l_cy),l_radius =cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy),r_radius =cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx,l_cy], dtype=np.int32)
            center_right = np.array([r_cx,r_cy], dtype=np.int32)

            # Calculate iris positions
            iris_pos_right, ratio_right = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            iris_pos_left, ratio_left = iris_position(center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0])
            
            # Append predicted iris positions to lists
            left_iris_positions.append(center_left)
            right_iris_positions.append(center_right)

                   
            # Calculate accuracy scores for right iris position
            if len(right_iris_positions) > 1:
                # Get the latest predicted position
                latest_right_iris_position = right_iris_positions[-1]
                # Calculate accuracy scores using the latest predicted position and ground truth
                mse_right = mean_squared_error([right_iris_ground_truth], [latest_right_iris_position])
                rmse_right = np.sqrt(mse_right)
                # Display MSE and RMSE scores for right iris position
            
                cv2.putText(frame, f"Right Iris MSE: {mse_right:.2f} RMSE: {rmse_right:.2f} ", (30, 120), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
            # Calculate accuracy scores for left iris position
            if len(left_iris_positions) > 1:
                latest_left_iris_position = left_iris_positions[-1]
                mse_left = mean_squared_error([left_iris_ground_truth], [latest_left_iris_position])
                rmse_left = np.sqrt(mse_left)
                # Display RMSE and MSE scores for left iris position
                cv2.putText(frame, f"Left Iris MSE: {mse_left:.2f} RMSE: {rmse_left:.2f} ", (30, 90), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv2.LINE_AA)    

            # Draw dot circles for left and right iris 
            cv2.circle(frame, center_left, 2, (0, 255, 0), 2)
            cv2.circle(frame, center_right, 2, (255, 0, 0), 2)

            #Draw white circles around the left and right eyeballs
            cv2.circle(frame,center_left,int(l_radius),(255,255,255),1,cv2.LINE_AA)
            cv2.circle(frame,center_right ,int(r_radius),(255,255,255),1,cv2.LINE_AA)
            
            # Draw reference points for the left and right ends of the eye 
            cv2.circle(frame,mesh_points[R_H_RIGHT][0] ,2,(255,255,255),1,cv2.LINE_AA)
            cv2.circle(frame,mesh_points[R_H_LEFT][0] ,2,(0,255,255),1,cv2.LINE_AA)
            cv2.circle(frame,mesh_points[L_H_RIGHT][0] ,2,(0,255,255),1,cv2.LINE_AA)
            cv2.circle(frame,mesh_points[L_H_LEFT][0] ,2,(0,255,255),1,cv2.LINE_AA)           
            
            # Calculate and display iris positions
            iris_pos_right,ratio_right=iris_position(center_right,mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
            iris_pos_left, ratio_left = iris_position(center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT][0])
            #print(iris_pos)
            cv2.putText(frame,f"Right Iris pos: {iris_pos_right} {ratio_right:.2f}" ,(30,30),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,0),1,cv2.LINE_AA)
           
            cv2.putText(frame, f"Left Iris pos: {iris_pos_left} {ratio_left:.2f}", (30, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
            
        # Display frame
        cv2.imshow('Iris Tracking Model',frame)
        key = cv2.waitKey(1)
        if key ==ord('q'):
            break
# Release video capture and close all windows
cap.release()    
cv2.destroyAllWindows()

