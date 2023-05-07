import os
from PIL import Image
import numpy as np


def resize_crop(image, target_hw=512):
    # Extract the original width and height of the image
    width, height = image.size
    
    # Calculate the scaling factor for each dimension
    width_factor = target_hw / width
    height_factor = target_hw / height
    
    # Choose the smaller scaling factor to ensure that the entire image fits within the maximum size
    scale_factor = max(width_factor, height_factor)
    
    # Calculate the new width and height based on the chosen scaling factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = image.resize((new_width, new_height))

    resized_arr = np.array(resized_image)
    if new_width <= new_height:
      images = [Image.fromarray(resized_arr[0:target_hw, ...]), 
                Image.fromarray(resized_arr[-target_hw:, ...])]
    else:
      images = [Image.fromarray(resized_arr[:, 0:target_hw, :]), 
                Image.fromarray(resized_arr[:, -target_hw:, :])]
    
    return images
  
  
def preprocess_data(directory_path, new_directory_path, target_size, new_extension='.png'):

  new_width, new_height = target_size

  # Create the new directory if it doesn't exist
  if not os.path.exists(new_directory_path):
      os.makedirs(new_directory_path)

  # Loop through all files in the directory
  for i, filename in enumerate(os.listdir(directory_path)):
      # Check if the file is an image by checking the extension
      if filename.split('.')[-1] in ('jpg', 'jpeg', 'png'):
          # Create a new filename by replacing the old extension with the new extension
          new_filename = os.path.basename(filename).split('.')[0]
          
          # Construct the full file paths
          old_file_path = os.path.join(directory_path, filename)
          new_file_path = os.path.join(new_directory_path, new_filename)
          
          # Load the image and convert it to RGB
          image = Image.open(old_file_path).convert("RGB")
          
          # Ignore Images with height or width less than 512
          if min(image.size) < 512: continue

          images = resize_crop(image)
          for j, img in enumerate(images):
            img.save(os.path.join(new_directory_path, str(i) + '_' + str(j) + new_extension))
