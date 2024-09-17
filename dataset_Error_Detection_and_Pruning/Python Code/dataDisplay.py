import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

def getImgData(image_name):
    hdf5 = h5py.File(hdf5_path, 'r')
    group_name = "bioscan_dataset"
    if group_name in hdf5.keys():
        hdf5 = hdf5[group_name]
    image = np.array(hdf5[image_name])
    image_data = Image.open(io.BytesIO(image))
    img_array = np.array(image_data)
    return img_array

hdf5_path = 'cropped_256.hdf5'
image_name = 'BIOUG71901-D05.jpg'
print(image_name)

img = getImgData(image_name)

plt.imshow(img)
plt.title(image_name)
plt.axis('off')

# Save the image to a file instead of showing it
plt.savefig("data_output_image.png", bbox_inches='tight')
