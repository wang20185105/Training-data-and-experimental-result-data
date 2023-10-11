import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
import imageio

Amazon_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/amazon'
Amazon_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/amazonprocess'
dlsr_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/dslr'
dlsr_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/dslrprocess'
webcam_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/webcam'
webcam_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/webcamprocess'
caltech_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/caltech'
caltech_procee_path = 'E:/pyworkspace/deep-coral-master/data/office_caltech_10/caltechprocess'

pathAmazonDir=os.listdir(Amazon_ImagePath)
pathdlsrDir=os.listdir(dlsr_ImagePath)
pathwebcamDir=os.listdir(webcam_ImagePath)
pathcaltechDir=os.listdir(caltech_ImagePath)
def image_resize(image_path, new_path):
    #print('===========>>reshape image size')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for img_name in os.listdir(image_path):
        img_path=image_path+'/'+img_name
        image=Image.open(img_path)
        image=image.resize((512,512))
        image.save(new_path+'/'+img_name)
    #print('end the processing!')
if __name__=='__main__':
    for cls in pathcaltechDir:
        image_resize(caltech_ImagePath+'/'+cls,caltech_procee_path+'/'+cls)
    R_channel=0
    G_channel=0
    B_channel=0
    num=0
    for cls in os.listdir(caltech_procee_path):
        for idx in range(len(cls)):
            filename=os.listdir(caltech_procee_path+'/'+cls)[idx]
            img=imageio.imread(os.path.join(caltech_procee_path+'/'+cls,filename))/255.0
            R_channel=R_channel+np.sum(img[:,:,0])
            G_channel=G_channel+np.sum(img[:,:,1])
            B_channel=B_channel+np.sum(img[:,:,2])
            num+=1
    num=num*512*512
    R_mean=R_channel/num
    G_mean=G_channel/num
    B_mean=B_channel/num
    print(R_mean)
    print(G_mean)
    print(B_mean)
    R_channel=0
    G_channel=0
    B_channel=0
    #num=0
    for cls in os.listdir(caltech_procee_path):
        for idx in range(len(cls)):
            filename=os.listdir(caltech_procee_path+'/'+cls)[idx]
            img=imageio.imread(os.path.join(caltech_procee_path+'/'+cls,filename))/255.0

            R_channel=R_channel+np.sum((img[:,:,0]-R_mean)**2)
            G_channel=G_channel+np.sum((img[:,:,1]-G_mean)**2)
            B_channel=B_channel+np.sum((img[:,:,2]-B_mean)**2)
    R_var=np.sqrt(R_channel/num)
    G_var=np.sqrt(G_channel/num)
    B_var=np.sqrt(B_channel/num)

    print(R_var)
    print(G_var)
    print(B_var)

