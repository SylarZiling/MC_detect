import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import torch

# pre-request class and functions 
class EulerAngles:
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    
    def __init__(self,roll,pitch,yaw):
        self.roll = roll
        self.pitch = pitch 
        self.yaw = yaw 

    def rad2deg(self):
        return np.array([self.pitch*180/np.pi, self.yaw*180/np.pi , (self.roll)*180/np.pi])
        #return np.array([self.pitch*180/np.pi, self.yaw*180/np.pi , (self.roll-np.pi/2)*180/np.pi])
   
class rotation_estimator:

    total_angles = EulerAngles(0,0,0)
    total_radians = EulerAngles(0,0,0)
    alpha = 0.98 
    firstGyro = True 
    firstAccel = True 
    # keeps the arrival time of previous gyro frame 
    first_ts_gyro = 0 

    def process_gyro(self,gyro_data,ts):
        if self.firstGyro: 
            self.firstGyro = False
            self.first_ts_gyro = ts
            return 

        gyro_angle = np.array([gyro_data.x,gyro_data.y,gyro_data.z])
        dt_gyro = (ts-self.first_ts_gyro)/1000.0
        self.first_ts_gyro = ts 

        gyro_angle = gyro_angle * dt_gyro

        #gyro_data.x : Yaw
        #gyro_data.y : Pitch 
        #gyro_data.z : Roll
        

        self.total_radians.yaw   -= gyro_angle[2]
        self.total_radians.roll  += gyro_angle[0]
        self.total_radians.pitch -= gyro_angle[1]


    def process_accel(self,accel_data):

        accel_anglez = np.arctan2(accel_data.y, accel_data.z)
        accel_anglex = np.arctan2(accel_data.x, np.sqrt((accel_data.y * accel_data.y) + (accel_data.z * accel_data.z)))

        if self.firstAccel:
            self.firstAccel = False 
            self.total_radians.yaw = accel_anglex
            self.total_radians.roll = accel_anglez
            self.total_radians.pitch = np.pi 
        else:
            self.total_radians.yaw = (self.total_radians.yaw * self.alpha) + (accel_anglex * (1-self.alpha))
            self.total_radians.roll = (self.total_radians.roll * self.alpha) + (accel_anglez * (1-self.alpha))


    def get_angles(self):
        #return an numpy array of pitch, yaw, roll in degrees 
        angles = np.array([self.total_radians.pitch*180/np.pi , 
                            self.total_radians.yaw*180/np.pi  , 
                            self.total_radians.roll*180/np.pi])
        
        return angles

    def get_radians(self):
        #returns an numpy array of pitch yaw roll in radians 
        angles = np.array([self.total_radians.pitch, 
                            self.total_radians.yaw, 
                            self.total_radians.roll])

        return angles

    def get_rotation_matrix(self):
        # return a np array of rotation matrix 
        rotationM = np.zeros((3,3))

        #a yaws beta roll 
        rotationM[0][0] = (np.cos(self.total_radians.yaw) * np.cos(self.total_radians.pitch))
        rotationM[0][1] = (np.cos(self.total_radians.yaw) * np.sin(self.total_radians.pitch) * np.sin(self.total_radians.roll) 
                            - np.sin(self.total_radians.yaw) * np.cos(self.total_radians.roll) )

        rotationM[0][2] = (np.cos(self.total_radians.yaw) * np.sin(self.total_radians.pitch) * np.cos(self.total_radians.roll) 
                            + np.sin(self.total_radians.yaw) * np.sin(self.total_radians.roll) )

        rotationM[1][0] = (np.sin(self.total_radians.yaw) * np.cos(self.total_radians.pitch))
        rotationM[1][1] = (np.sin(self.total_radians.yaw) * np.sin(self.total_radians.pitch) * np.sin(self.total_radians.roll) 
                            + np.cos(self.total_radians.yaw) * np.cos(self.total_radians.roll) )
        rotationM[1][2] = (np.sin(self.total_radians.yaw) * np.sin(self.total_radians.pitch) * np.cos(self.total_radians.roll) 
                            - np.cos(self.total_radians.yaw) * np.sin(self.total_radians.roll) )

        rotationM[2][0] = -np.sin(self.total_radians.pitch)
        rotationM[2][1] = (np.cos(self.total_radians.pitch) * np.sin(self.total_radians.roll))
        rotationM[2][2] = (np.cos(self.total_radians.pitch) * np.cos(self.total_radians.roll))

        return rotationM

    def get_pitch_matrix(self):
        rotationM = np.zeros((3,3))
        # pitch rotation matrix 
        rotationM[0][0] = np.cos(self.total_radians.pitch)
        rotationM[0][2] = np.sin(self.total_radians.pitch)
        rotationM[1][1] = 1
        rotationM[2][0] = -np.sin(self.total_radians.pitch)
        rotationM[2][2] = np.cos(self.total_radians.pitch)
        return rotationM

    def get_roll_matrix(self):
        rotationM = np.zeros((3,3))
        # roll rotation matrix 
        rotationM[0][0] = 1
        rotationM[1][1] = np.cos(self.total_radians.roll)
        rotationM[1][2] = -np.sin(self.total_radians.roll)
        rotationM[2][1] = np.sin(self.total_radians.roll)
        rotationM[2][2] = np.cos(self.total_radians.roll)
        return rotationM

    def get_yaw_matrix(self):
        rotationM = np.zeros((3,3))
        # yaw rotation matrix 
        rotationM[0][0] = np.cos(self.total_radians.yaw)
        rotationM[0][1] = -np.sin(self.total_radians.yaw)
        rotationM[1][0] = np.sin(self.total_radians.yaw)
        rotationM[1][1]= np.cos(self.total_radians.yaw)
        rotationM[2][2] = 1
        return rotationM

#==================================================================================================
# functions for the process 
def takeFrame(device_name,mode_name):
    # Enable Setup: 
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
    profile = pipe.start(cfg)

    #set depth sonsor 
    depth_sensor = profile.get_device().first_depth_sensor()
    preset_range = depth_sensor.get_option_range(rs.option.visual_preset)

    #set camera to preset no Ambient Light to improve the depth qualiy for L515
    for i in range(int(preset_range.max)):
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
        print('%02d: %s'%(i,visulpreset))
        if device_name == "L515":
            if visulpreset == mode_name:#"No Ambient Light":# L515 
                depth_sensor.set_option(rs.option.visual_preset, i)
        if device_name == "D435":
            if visulpreset == mode_name:#"High Density":# D435    
                depth_sensor.set_option(rs.option.visual_preset, i)

    #estimation -> take data 
    estimation = rotation_estimator()

    for x in range(5):
        pipe.wait_for_frames()
        
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    gyro = frameset[3].as_motion_frame().get_motion_data()
    accel = frameset[2].as_motion_frame().get_motion_data()

    ts = frameset.get_timestamp()
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()
    pipe.stop()
    colorizer = rs.colorizer()
    color = np.asanyarray(color_frame.get_data())
    #aligned_depth_img = np.asanyarray(aligned_depth_frame.get_data())
    aligned_colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    return color,aligned_depth_frame,gyro,accel,ts,estimation 

def get_matrix_intrin(aligned_depth_frame,estimation,gyro,accel,ts):
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    estimation.process_gyro(gyro,ts)
    estimation.process_accel(accel)

    matrix = estimation.get_rotation_matrix()
    roll_matrix = estimation.get_roll_matrix()
    pitch_matrix = estimation.get_pitch_matrix()
    yaw_matrix = estimation.get_yaw_matrix()
    
    angle = estimation.get_angles()

    radians = estimation.get_radians()
    print("radians is : ")
    print(radians)

    print(" ")
    print("general angle is: pitch yaw roll ")
    print(angle)

    return depth_intrin,matrix 

def getArea(a,b,c,d):
    return abs(c-a) * abs(d-b)

def objectDetection(model,image,obj_name):
    # detect the cup with the model yolov5
    results = model(image)
    boxs= results.pandas().xyxy[0].values

    cup_box = np.zeros((4,1))
    cup_boxes = np.array([[0,0,0,0]])
    maxArea = 0 
    for box in boxs:
        print(box[-1])
        cup_box = np.zeros((5,1))
        if box[-1] == obj_name:
            area = getArea(box[0],box[1],box[2],box[3])
            if area > maxArea: 
                cup_box[0] = box[0]
                cup_box[1] = box[1]
                cup_box[2] = box[2]
                cup_box[3] = box[3]
            #cup_box[4] = int() * int()
                cup_boxes = np.insert(cup_boxes,0,cup_box, axis = 0)

    if len(cup_boxes) == 1:
        print("did not found the required objects")
        return 
    else:
        print("found object at : ")
        mid_pos = [int((cup_box[0] + cup_box[2])//2) , int((cup_box[1] + cup_box[3])//2)]
        mid_pos = np.array(mid_pos)
        print(mid_pos)
        return mid_pos,cup_box,boxs

def getDepth(aligned_depth_frame,pos,depth_intrin):
    
    mid_depth = aligned_depth_frame.get_distance(pos[0],pos[1]) 

    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,pos,mid_depth)

    return depth_point,mid_depth 

def applyWrap(image,estimation):
    height,width = image.shape[:2]
    center_x,center_y = (width/2,height/2)
    
    allang = estimation.total_radians.rad2deg()
    print(allang)
    
    M = cv2.getRotationMatrix2D((center_x,center_y),allang[1],1.0) # rotation from original to flat
    inverM = cv2.invertAffineTransform(M)                          # rotation from flat to original 

    rotated_img = cv2.warpAffine(image,M,(width,height))

    return M,inverM,rotated_img,allang

def transferpoints(x,y,mat):
    points = np.array([[x],[y],[1]])
    results = mat @ points
    return results

def get_num(points):
    # return the 3D ray away from the camera center 
    return np.sqrt(points[0]*points[0] + points[1]*points[1] + points[2]*points[2])

def get_distance(points1,points2):
    #return the distane between two 3d points in depth frame 
    return np.sqrt((points2[0] - points1[0])**2.0 + (points2[1] - points1[1])**2.0 + (points2[2] - points1[2])**2.0)

def getCenter(tvalue,cup_box,cup_boxN,image,mid_posN,wrapmid,depth_intrin,matrix,M ,aligned_depth_frame):
    far_pt = []
    far_3d = wrapmid

    close_pt = []
    close_3d = wrapmid

    bottom_pt = []
    bottom_3d = wrapmid
    #x,y,z pose 
    toplist = [[],[],[],[]]
    verlist = [[],[],[],[]]
    # find all minimum z to find the rough tsize 
    # find the top and bottom and furest 
    for x in range(640):
        for y in range(480):
            #if it is inside the depth frame box 
            if cup_box[0]<=x<cup_box[2] and cup_box[1]<=y<cup_box[3]:
                # if it is not black 
                if (image[y][x] != [0,0,0]).any():
                    #transfer the point into the flat image and see if it is in the box 
                    wrapX,wrapY = transferpoints(x,y,M)
                    if cup_boxN[0] <= wrapX <=cup_boxN[2] and cup_boxN[1] <= wrapY <= cup_boxN[3]:
                        # if the wrapped points is in the cup box 
                        depth = aligned_depth_frame.get_distance(x,y) 
                    
                        # if depth<tvalue[0] or depth>tvalue[1]:
                        #     #if in box but it is too far away or too close
                        #     image[y][x] = [0,0,0] 
                        if tvalue[0] < depth < tvalue[1]:
                        #else:
                            #if in box but it is too far away or too close
                            pose = np.array([x,y])                                  # position in original image 
                            #flatpose = np.array([int(wrapX),int(wrapY)])            # position in wrapped image 
                            points = rs.rs2_deproject_pixel_to_point(depth_intrin,pose,depth) # 3Dpts in depth F
                            #length = get_num(points)                                # length of ray to the point in depth F
                            wrapPoint = matrix @ points                             # 3Dpts in original F 
                            #wraplen = get_num(wrapPoint)

                            # Find the most close point first 
                            if wrapPoint[1] <close_3d[1] and wrapY <= (cup_boxN[1]+30) and wrapPoint[2] < close_3d[2]:
                                close_pt = pose 
                                close_3d = wrapPoint

                            #if wrapPoint[2] <= far_3d[2] and wrapPoint[1] >= far_3d[1]:
                            if abs(wrapPoint[2]-close_3d[2]) <= 0.01 and wrapY <= (cup_boxN[1]+30) and wrapPoint[1] > far_3d[1]:
                                far_3d = wrapPoint
                                far_pt = pose

                            if abs(wrapPoint[1] - wrapmid[1]) <= 0.005 and abs(wrapX - mid_posN[0]) <= 20 and wrapPoint[2] > bottom_3d[2]:
                                bottom_pt = pose
                                bottom_3d = wrapPoint
                                
                            # use the limitation with cup_boxN, if the size is not too far away from cup boundary y -> store points find z 
                            if wrapY <= (cup_boxN[1]+30): 
                                toplist[0].append(wrapPoint[0])
                                toplist[1].append(wrapPoint[1])
                                toplist[2].append(wrapPoint[2])
                                toplist[3].append(pose)
                            
                            if abs(wrapX-mid_posN[0]) <= 5 :
                                verlist[0].append(wrapPoint[0])
                                verlist[1].append(wrapPoint[1])
                                verlist[2].append(wrapPoint[2])
                                verlist[3].append(pose)
                        # else: 
                        #     image[y][x] = [0,0,0]
            #         else:           
            #             image[y][x] = [0,0,0]        
            # else:
            #     image[y][x] = [0,0,0] 

    diameter = (max(toplist[1]) - min(toplist[1]) + max(toplist[0]) - min(toplist[0]) )/2
    diameter2 = abs(close_3d[1]- far_3d[1])
    height2 = bottom_3d[2] - close_3d[2]
    height = max(verlist[2]) - min(verlist[2])

    middle_y = wrapmid[1]- diameter
    middle_z = wrapmid[2] + (height)/2
    middle_x = wrapmid[0]
    print("mid in the y axis is: " + str(middle_y))
    print("mid in the z axis is: " + str(middle_z))
    print("mid in the x axis is: " + str(middle_x))
    middle = np.array([wrapmid[0],wrapmid[1]- diameter,wrapmid[2] + (height)/2])
    return  middle, diameter, height 

def main():
    # import the modules 
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
    model.conf = 0.5

    # set up the pipeline and take the frame 
    color,aligned_depth_frame,gyro,accel,ts,estimation = takeFrame("D435","High Density")
    # D435                  ====    L515 
    # 00: Custom                    00: Custom
    # 01: Default                   01: Default
    # 02: Hand                      02: 
    # 03: High Accuracy             03: 
    # 04: High Density              04: No Ambient Light 

    # detect the mid pose, cup_box and all boxes 
    mid_pos,cup_box,boxs = objectDetection(model,color,'cup')

    # get matrix and camera intrinsic 
    depth_intrin,matrix = get_matrix_intrin(aligned_depth_frame,estimation,gyro,accel,ts)

    # use intrinsic to get depth of the cup 
    depth_point,mid_depth = getDepth(aligned_depth_frame,mid_pos,depth_intrin)

    print("3d point is : ")
    print(depth_point)
    # get rotated image and matrix , inverse matrix 
    M,inverM,rotated_img,allang = applyWrap(color,estimation)

    # detect the mid pose in rotated image    
    mid_posN,cup_boxN,boxesN = objectDetection(model,rotated_img,'cup')

    tvalue = [mid_depth-0.01,mid_depth+0.1]# assume the furthest point is 10 cm and close is 1cm 
    wrapmid = matrix@depth_point 
    print("wrapped 3d point is ")
    print(wrapmid)
    middle, diameter, height =getCenter(tvalue,cup_box,cup_boxN,color,mid_posN,wrapmid,depth_intrin,matrix,M ,aligned_depth_frame)
    print(middle)
    print(diameter)
    print(height)
    

if __name__ == "__main__":
    main()
    