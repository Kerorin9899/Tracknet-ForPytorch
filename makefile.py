#2.Output training data name to cvs file for model 1
import numpy as np
import cv2
import glob
import itertools
import random
import csv
import pdb

import argparse


training_file_name = "csv/Tracknet_Ftr.csv"
visibility_for_testing = []

allimages = []

#Test [13, 14, 27, 37, 38, 53, 57, 68, 79, 89]

test_number = [13, 14, 27, 37, 38, 53, 57, 68, 79, 89]

for clip in range(14,22):

    #print(clip)
    if clip in test_number:
       continue

    #################change the path####################################################
    images_path = "DataClip/Dataimages/Clip" + str(clip) + "/"
    annos_path = "GroundTruth/Clip" + str(clip) + "/"
    prop_path = "PropImages/Pre150/Clip" + str(clip) + "/"
    ####################################################################################

    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    props = glob.glob( prop_path + "*.jpg"  ) + glob.glob( prop_path + "*.png"  ) +  glob.glob( prop_path + "*.jpeg"  )
    props.sort()
    annotations  = glob.glob( annos_path + "*.jpg"  ) + glob.glob( annos_path + "*.png"  ) +  glob.glob( annos_path + "*.jpeg"  )
    annotations.sort()
        

    visibility = {}
    x_coordinate = {}
    y_coordinate = {}

    with open(images_path + "Label.csv", 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #skip the headers
        next(spamreader, None)
        #pdb.set_trace()
        
        j = 0
        for row in spamreader:
            #row[0] => image name
            #row[1] => visibility class
            visibility[j] = row[1]
            x_coordinate[j] = row[2]
            y_coordinate[j] = row[3]
            j += 1
            #pdb.set_trace()

    #write all of images path and distance
    for i in range(2,len(images)-1): 
        allimages.append([images[i+1],images[i],images[i-1],annotations[i]])

        if visibility[i] == "0":
            continue
        #allimages.append([images[i],images[i-1],images[i-2],x_coordinate[i], y_coordinate[i]])

print(len(allimages))

random.shuffle(allimages)

with open(training_file_name,'w', newline="") as file:
    for i in range(len(allimages)):
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow([allimages[i][0],allimages[i][1],allimages[i][2],allimages[i][3]])
file.close()

#read all of images path
lines = open(training_file_name).read().splitlines()
print("Total images:", len(lines))
