import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import glob
import sklearn
from extract_frames import frameTesting
from extract_frames import allFramesTesting



def predict():
   # Extracting testing video frames
   frameTesting()
   # Pass the path of the Frames with video names attached
   first_path = "C:\\Users\\Ishdeep Bhandari\\Desktop\\PythonApplication1\\PythonApplication1\\testing_data\\frames"
   path  = os.path.join(first_path,'*g')
   files = glob.glob(path)
   image_size= 128
   num_channels=3  
   for filename in files:
       images = []
  # Reading the image 
       image = cv2.imread(filename)
  # Resizing the image to the desired size and preprocessing done exactly as done during training
       image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
       images.append(image)
       images = np.array(images, dtype=np.uint8)
       images = images.astype('float32')
       images = np.multiply(images, 1.0/255.0) 

 # The input to the network is of shape [None image_size image_size num_channels]. So reshaping as per required
       x_batch = images.reshape(1, image_size,image_size,num_channels)

 # Restore the saved model 
       sess = tf.Session()
 # Recreate the network graph 
       saver = tf.train.import_meta_graph('action-classifier-model.meta')
 # Load the weights saved 
       saver.restore(sess, tf.train.latest_checkpoint('./'))

 # Accessing the default graph which was restored
       graph = tf.get_default_graph()

 # y_pred used in the original network
       y_pred = graph.get_tensor_by_name("y_pred:0")

 # Feed the images to the input placeholders
       x= graph.get_tensor_by_name("x:0") 
       y_true = graph.get_tensor_by_name("y_true:0") 
       y_test_images = np.zeros((1, 5)) 
 

# Creating the feed_dict that is required to be fed to calculate y_pred 
# Return the frame of the video with the classification
       feed_dict_testing = {x: x_batch, y_true: y_test_images}
       result=sess.run(y_pred, feed_dict=feed_dict_testing)
       Saved_Result = 'For ',os.path.basename(filename), ':', '\n', 'JumpingJack: %f \n' % result[0][0],'Lunges: %f \n' % result[0][1],'WallPushups: %f \n' % result[0][2], 'BrushingTeeth: %f \n' % result[0][3],'CuttinginKitchen: %f \n \n' % result[0][4]
       with open('Predictions','a') as a:
          a.write("".join(Saved_Result))
       print('For ',os.path.basename(filename), ':', '\n', 'JumpingJack: %f \n' % result[0][0],'Lunges: %f \n' % result[0][1],'WallPushups: %f \n' % result[0][2], 'BrushingTeeth: %f \n' % result[0][3],'CuttinginKitchen: %f \n' % result[0][4])
     

   
    





