import cv2
from keras.models import load_model
#from keras.utils.generic_utils import CustomObjectScope
import glob
#from models.unets import Unet2D
from modells.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from modells.FCN import FCN_Vgg16_16s

from utils.learning.metrics import dice_coef, precision, recall
#from utils.BilinearUpSampling import BilinearUpSampling2D
from utils.io.data import load_data, save_results, save_rgb_results, save_history, load_test_images, DataGen

import numpy as np

import mysql.connector

# Connect to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="npwt"
)

# Get a cursor
mycursor = mydb.cursor()

# Get the file paths from the database
mycursor.execute("SELECT `before_wound`,`after_wound` FROM `image` ORDER BY `image`.`time_stamp` DESC limit 1 ")
result = mycursor.fetchone()
file1_path = result[0]
file2_path = result[1]

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'

path = './data/Medetec_foot_ulcer/'

weight_file_name = 'model.hdf5'

pred_save_path = 'predictions/'

read_path = "data/Medetec_foot_ulcer/test/predictions\*.*"

file1 = "data/Medetec_foot_ulcer/test/predictions/before.png"
file2 = "data/Medetec_foot_ulcer/test/predictions/after.png"

# file1=file1_path
# file2=file2_path

data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
x_test, test_label_filenames_list = load_test_images(path)

model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model = load_model('./training_history/' + weight_file_name
               , custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef,
                                 'relu6':relu6,
                                 'DepthwiseConv2D':DepthwiseConv2D,
                                 'BilinearUpsampling':BilinearUpsampling})

for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
    prediction = model.predict(image_batch, verbose=1)
    save_results(prediction, 'rgb', path + 'test/' + pred_save_path, test_label_filenames_list)
    #for file in glob.glob(read_path):
    img1 = cv2.imread(file1)
    n_white_pix1 = np.sum(img1 == 255)
    n_white_pix1 = n_white_pix1/100
    if n_white_pix1>100:
          n_white_pix1=100
          print('Wound in Percentage:', 100,'%')
    else:
         print('Wound in Percentage:', n_white_pix1,'%')
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    c1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    c1 = cv2.putText(img1, str(n_white_pix1)+' %', org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Before wound image', c1)
    cv2.waitKey(3000)

    img2 = cv2.imread(file2)
    n_white_pix2 = np.sum(img2 == 255)
    n_white_pix2 = n_white_pix2/100
    print('Wound in Percentage:', n_white_pix2,'%')
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (30, 30)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    c2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    c2 = cv2.putText(img2, str(n_white_pix2)+' %', org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('After wound image', c2)
    cv2.waitKey(3000)
    wound_healing = n_white_pix1 - n_white_pix2
    print('Wound Healing in Percentage:', wound_healing,'%')
    # Insert the data into the database
    sql = "INSERT INTO wound_prediction (before_percentage, after_percentage, healing_percentage) VALUES (%s, %s, %s)"
    val = (n_white_pix1, n_white_pix2, wound_healing)
    mycursor.execute(sql, val)

# Commit the changes
    mydb.commit()

# Close the database connection
    mydb.close()
 
     
    break

