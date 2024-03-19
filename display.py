import numpy as np
import matplotlib.pyplot as plt
import math

if __name__=="__main__":
        # Assuming you have pixel values stored in a 2D NumPy array
    with open('output.txt', 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.split())) for line in lines]
    data_array = np.array(data)
    data_array_reshaped = data_array.reshape(64, 64)
    rotated_data_array = np.rot90(data_array_reshaped, k=-1)
    

    scaled_rotated_data_array = (rotated_data_array - np.min(rotated_data_array)) * 255 / (np.max(rotated_data_array) - np.min(rotated_data_array))
    print(scaled_rotated_data_array)  
    # Plot the image
    plt.imshow(scaled_rotated_data_array, cmap='gray', vmin=0, vmax=255)  # Use 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis
    plt.show()
