import cv2
import math
import os
import glob

classes = ['JumpingJack','Lunges','WallPushups','BrushingTeeth','CuttingInKitchen']

# Extracts all frames of videos required for training
def allFramesTraining():
    print("Extracting frames")
    # Path to the raw videos for training 
    first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\data\\ucf-101"
    for fields in classes:
        # Specify what type of file to get .avi or .mp4
        path  = os.path.join(first_path,fields,'*.avi')
        files = glob.glob(path)
        for filename in files:
            vidcap = cv2.VideoCapture(filename)
            success,image = vidcap.read()
            success = True
            actual_filename = os.path.basename(filename)
            count = 1
            while success:
                # Extracted frames saved to the folder as the same name as the action in the training_data folder
                img_path = 'C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\training_data'
                cv2.imwrite(os.path.join(img_path, fields, '%s %d.jpg' %(actual_filename, count)), image)
                success,image = vidcap.read()
                print('Read a new frame: ', success, actual_filename)
                count += 1
        count = 1
        print("Finished extracting frames")

# Extracts all frames of videos required for testing predictions
def allFramesTesting():
    print("Extracting frames")
    # Path to the testing videos
    first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\data\\own" 
   # Speicfy what type of file to get .avi or .mp4
    path  = os.path.join(first_path,'*.mp4')
    files = glob.glob(path)
    for filename in files:
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        success = True
        actual_filename = os.path.basename(filename)
        count = 1
        while success:
            # Extracted frames saved to this folder required for testing
            img_path = 'C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\testing_data\\frames'
            cv2.imwrite(os.path.join(img_path, '%s %d.jpg' %(actual_filename, count)), image)
            success,image = vidcap.read()
            print('Read a new frame: ', success, actual_filename)
            count += 1
    count = 1
    print("Finished extracting frames")


 # Extracts middle frames of videos and flips them to double the dataset; required for training
def frameTesting():
        print("Extracting frames")
        # Path to the testing videos
        first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\data\\own" 
        # Speicfy what type of file to get .avi or .mp4
        path  = os.path.join(first_path,'*.mp4') 
        files = glob.glob(path)
        for filename in files:
            vidcap = cv2.VideoCapture(filename)
            actual_filename = os.path.basename(filename)
            Frame_no = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            vidcap.set(1,int(Frame_no/2))
            res,image = vidcap.read()
            # Extracted frames saved to this folder required for testing
            img_path = 'C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\testing_data\\frames'
            cv2.imwrite(os.path.join(img_path, '%s.jpg' % actual_filename), image)
            flip_image = cv2.flip(image,1)
            cv2.imwrite(os.path.join(img_path, '%s_Flip.jpg' % actual_filename), flip_image)
        print("Finished extracting frames")

 # Extracts middle frames of videos and flips them to double the dataset; required for testing predictions
def frameTraining():
        print("Extracting frames")
        # Path to the raw videos for training
        first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\data\\ucf-101" 
        for fields in classes:
            # Speicfy what type of file to get .avi or .mp4
            path  = os.path.join(first_path,fields,'*.avi')
            files = glob.glob(path)
            for filename in files:
                vidcap = cv2.VideoCapture(filename)
                actual_filename = os.path.basename(filename)
                Frame_no = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                vidcap.set(1,int(Frame_no/2))
                res,image = vidcap.read()
                # Extracted frames saved to the folder as the same name as the action in the training_data folder
                img_path = 'C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\training_data'
                cv2.imwrite(os.path.join(img_path,fields, '%s.jpg' % actual_filename), image)
                flip_image = cv2.flip(image,1)
                cv2.imwrite(os.path.join(img_path,fields, '%s_Flip.jpg' % actual_filename), flip_image)
        print("Finished extracting frames")

 # Extracts 3 frames namely; the middle, the flip of the middle and the middle+4th frame
def multipleFrameTraining():
        print("Extracting frames")
        # Path to the raw videos for training
        first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\data\\ucf-101" 
        for fields in classes:
            # Speicfy what type of file to get .avi or .mp4
            path  = os.path.join(first_path,fields,'*.avi')
            files = glob.glob(path)
            for filename in files:
                vidcap = cv2.VideoCapture(filename)
                actual_filename = os.path.basename(filename)
                Frame_no = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                vidcap.set(1,int(Frame_no/2))
                res,image = vidcap.read()
                # Saves the file in the folder as the same name as the action in the training_data folder
                img_path = 'C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\training_data'
                cv2.imwrite(os.path.join(img_path,fields, '%s.jpg' % actual_filename), image)
                flip_image = cv2.flip(image,1)
                # Flipped middle Frame
                cv2.imwrite(os.path.join(img_path,fields, '%s_Flip.jpg' % actual_filename), flip_image)
                vidcap.set(1,int(Frame_no/2)+4)
                res,image = vidcap.read()
                # Handpicked frame 4 frames after the middle frame
                cv2.imwrite(os.path.join(img_path,fields, '%s_HandPicked.jpg' % actual_filename), image)

        print("Finished extracting frames")


def extractFrames():
    print("Extracting frames")
    # Path to the raw videos for training
    files = "C:\\Users\\Ishdeep Bhandari\\Downloads\\BaseY.mp4"
    vidcap = cv2.VideoCapture(files)
    actual_filename = os.path.basename(files)
    Frame_no = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 1
    while True:
        vidcap.set(1,i)
        res, image = vidcap.read()
        # Extracted frames saved to the folder as the same name as the action in the training_data folder
        img_path = 'C:\\Users\\Ishdeep Bhandari\\Downloads\\Bases\\BaseY'
        cv2.imwrite(os.path.join(img_path,'%s %d.jpg' % (actual_filename,i)), image)
        i = i+5
        if (i >= int(Frame_no)):
            break
    print("Finished extracting frames")

extractFrames()
