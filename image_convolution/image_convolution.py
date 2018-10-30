import numpy as np
import scipy.signal
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def Convolution(img,filt):
	m1,m2 = img.shape
	n1,n2 = filt.shape

	y = np.zeros((m1,m2))

	for i in range(m1):
		for j in range(m2):
			for ii in range(n1):
				for jj in range(n2):
					if i-ii>=0 and j-jj>=0:
						y[i,j]+=img[i-ii,j-jj]*filt[ii,jj]

	return y

img = mpimg.imread('Lena.png')
## img.shape 
##(512,512,3)

### Finding the mean across all the RGB channels
img_mean = img.mean(axis=2)
## img_mean.shape
## (512,512)



### Vertical edge Detector
Hx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
##
## Hx.shape
## 3,3
Hy = Hx.T 


plt.imshow(img_mean)
plt.title('Normalized Image')
plt.show()
plt.savefig('Normalized_Image')

Gx = Convolution(img_mean,Hx)
plt.imshow(Gx)
plt.title('Vertical Filter Output')
plt.show()
plt.savefig('Vertical_Filter_Output')

Gy = Convolution(img_mean,Hy)
plt.imshow(Gy)
plt.title('Horizontal Filter Output')
plt.show()
plt.savefig('Horizontal_Filter_Output')

G = np.sqrt(Gx*Gx+Gy*Gy)

plt.imshow(G)
plt.title('Resultant Filter Output')
plt.show()
plt.savefig('Resultant_Filter_Output')