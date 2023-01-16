#Model creation
from tensorflow import keras
from keras.layers import Input, Conv2D
from keras.models import Model
from sklearn.model_selection import train_test_split

#path sorting
import glob
from pathlib import Path
import re
import os

#math
import numpy as np
import matplotlib.pyplot as plt
import math

#image stuff
import cv2
from PIL import Image
from patchify import patchify, unpatchify

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
		match = file_pattern.match(Path(file).name)
		if not match:
		    return math.inf
		return int(match.groups()[0])


def run_model(model, testim_path, out_path, SIZE_X, SIZE_Y): 
	filename = testim_path.split('/')[-1].split(".")[0]

	start_img = cv2.imread(testim_path, cv2.IMREAD_UNCHANGED)
	test_img = cv2.resize(start_img, (SIZE_X, SIZE_Y))
	test_img = np.expand_dims(test_img, axis=0)

	prediction = model.predict(test_img)
	
    #View and Save segmented image
	
	prediction_image = prediction.reshape((test_img.shape[1], test_img.shape[2]))
	plt.imsave(f'{out_path}/{filename}.png', prediction_image, cmap='gray')
	resizing = cv2.imread(f'{out_path}/{filename}.png', cv2.IMREAD_GRAYSCALE)
	resizing = cv2.resize(resizing, (244, 244))
	plt.imsave(f'{out_path}/{filename}.png', resizing, cmap='gray')


# creates a model instance, and runs this model on the patches
# Then places these results in the patches_results folder
def model_on_patches(SIZE_X, SIZE_Y, block_results_folder):


	if not os.path.exists(f"{block_results_folder}patches_results"):
		os.mkdir(f"{block_results_folder}patches_results")


	out_path = f"{block_results_folder}patches_results/"
	patches_folder = f"{block_results_folder}patches/"

	model = keras.models.load_model(f"model/cloud_399im_100e.h5", compile=False)

	print("Applying model to patches...")
	for patch_file in sorted(glob.glob(patches_folder+"*.png"), key=get_order):
		print(patch_file)
		run_model(model, patch_file, out_path, SIZE_X, SIZE_Y)
	print("Finished applying model to patches")


## Alrighty retard its patchify time
## you are going to have to grab each image and turn it into a numpy array, then unpatchify that list of arrays. Try maybe one row first?
def reconstruct_patch_results(block_results_folder):


	patches = []
	patches_row = []

	for patch_pred_file in sorted(glob.glob(block_results_folder+"patches_results/"+"*.png"), key=get_order):
		patch = cv2.imread(patch_pred_file, cv2.IMREAD_GRAYSCALE)
		patch = np.asarray(patch)
		patches_row.append(patch)
		if len(patches_row) == 15:
			patches.append(np.array(patches_row))
			patches_row = []

	patches = np.array(patches, dtype=object)




	reconstructed_image = unpatchify(patches, (3660, 3660))
	reconstructed_image = Image.fromarray(np.uint8(reconstructed_image))
	reconstructed_image.save(block_results_folder+"combined_result.png")



