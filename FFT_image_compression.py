import numpy as np
import matplotlib.pyplot as plt

#read the image into a numpy array and convert it into a gray scale image 

img=plt.imread("/content/drive/MyDrive/pexels-arsham-haghani-3445219.jpg")

img_gray=0.299*img[:,:,0]+0.59*img[:,:,1]+0.11*img[:,:,2]
plt.imshow(img_gray,cmap="gray")

img_fft=np.fft.fft2(img_gray)
thre_img=img_fft.copy()

img_fft_abs_flatten=abs(img_fft.flatten())
img_fft_abs_flatten_sort=np.sort(img_fft_abs_flatten)

for per in [0.1,0.05,0.02,0.002,0.0002]:
  thres=img_fft_abs_flatten_sort[int(img_fft_abs_flatten_sort.shape[0]*((1-per)))]
  print("threshold: ",thres)
  for i in range(img_fft.shape[0]):
    for j in range(img_fft.shape[1]):
      if(abs(img_fft[i,j])<thres):
        thre_img[i,j]=0


  inv_threimg=np.fft.ifft2(thre_img).real
  plt.title("compressed image by keeping {} % of image".format(per*100))
  plt.imshow(inv_threimg,cmap="gray")
  plt.show()