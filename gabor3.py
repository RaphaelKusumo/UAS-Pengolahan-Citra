#import library
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
import scipy.ndimage as ndi

# prepare filter bank kernels
kernels = []
for theta in range(4):
 theta = theta / 4. * np.pi
 for sigma in (1, 3):
  for frequency in (0.05, 0.25):
   kernel = np.real(gabor_kernel(frequency, \
 theta=theta, sigma_x=sigma, sigma_y=sigma))
 kernels.append(kernel)

def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

#load reference image                 
image_names = ['images/UIUC_textures/woods/T04_01.jpg',
 'images/UIUC_textures/stones/T12_01.jpg',
 'images/UIUC_textures/bricks/T15_01.jpg',
 'images/UIUC_textures/checks/T25_01.jpg']
labels = ['woods', 'stones', 'bricks', 'checks']

images = []
for image_name in image_names:
 images.append(rgb2gray(imread(image_name)))

#Create four filter bank kernels with different values of parameters (theta and frequency):
results = []
kernel_params = []
for theta in (0, 1):
 theta = theta / 4. * np.pi
 for frequency in (0.1, 0.4):
  kernel = gabor_kernel(frequency, theta=theta)
 params = 'theta=%d,\nfrequency=%.2f' % \
    (theta * 180 / np.pi, frequency)
 kernel_params.append(params)
 results.append((kernel, [power(img, kernel) for img \
 in images]))

 #function to extract the features of an image corresponding to the Gabor filter bank kernels
def compute_feats(image, kernels):
 feats = np.zeros((len(kernels), 2), dtype=np.double)
 for k, kernel in enumerate(kernels):
  filtered = ndi.convolve(image, kernel, mode='wrap')
 feats[k, 0] = filtered.mean()
 feats[k, 1] = filtered.var()
 return feats

#Implement a function match that performs the classification task
def match(feats, ref_feats):
 min_error = np.inf
 min_i = None
 for i in range(ref_feats.shape[0]):
  error = np.sum((feats - ref_feats[i, :])**2)
 if error < min_error:
  min_error = error
 min_i = i
 return min_i

# tract the reference image's features and the new test image's features. Classify the test images—that is, match each new (test) image with the nearest reference image and label the test image with the class of the reference image:
ref_feats = np.zeros((4, len(kernels), 2), dtype=np.double)
for i in range(4):
 ref_feats[i, :, :] = compute_feats(images[i], kernels)
print('Images matched against references using Gabor filter banks:')
new_image_names = ['images/UIUC_textures/woods/T04_02.jpg',
 'images/UIUC_textures/stones/T12_02.jpg',
 'images/UIUC_textures/bricks/T15_02.jpg',
 'images/UIUC_textures/checks/T25_02.jpg',
 ]
for i in range(4):
 image = rgb2gray(imread(new_image_names[i]))
 feats = compute_feats(image, kernels)
 mindex = match(feats, ref_feats)
 print('original: {}, match result: {}'.format(labels[i],
 labels[mindex]))

#plot awal
#untuk 3 dan 4
 results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))
 
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(6, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel))
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()


