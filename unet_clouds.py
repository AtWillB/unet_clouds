#Will Byrne January 10th, 2023

import numpy as np
import matplotlib.pyplot as plt
import model_application as mod_app

from patchify import patchify, unpatchify
import cv2
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import glob
import os

import osgeo_utils.gdal_merge as merge
from osgeo import gdal




# Grab the bands from the specified block, and create a dictionary that stores the path to 
# all of the blocks bands, and the blocks name.
# Return the dictionary
def get_bands(block_path):
    block_dict = {}
    block_name = block_path.split("/")[-2]
    block_dict.update({"block_name": block_name})

    for band in glob.glob(block_path+"*B[0-9][A-Z0-9].tif"):
        band_name = band.split(".")[-2]
        block_dict.update({band_name: band})
        
    return block_dict


# Create a folder dedicated to this block. Create a merged image of the selected bands using gdal_merge.py. 
# Place this "merged.tif" file in the dedicated block folder. 
# Return the path to the folder for this block. 
def make_image(block_dict, band_names):
    results_path = 'unet_results/'+block_dict['block_name']+"/"
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    image_band_list = []
    for band in band_names:
        image_band_list.append(block_dict[band])

    print(f"Merging {band_names} from {block_dict['block_name']}: ")
    parameters = ['', '-o', results_path+"/merged.tiff"] + image_band_list + ['-separate']
    merge.main(parameters)
    print("\n")

    return results_path


# Create a png of the geotiff(program was trained on pngs, but conversion may be unessesary)
# Devide the merged image into patches, and create a folder for those divided patches. 
# Returns the path to the patches folder
def patchify_image(results_path):
    patches_folder = results_path+"patches"
    if not os.path.exists(patches_folder):
        os.mkdir(patches_folder)


    ds = gdal.Open(results_path+"merged.tiff")
    gdal.Translate(results_path+"merged.png", ds, format = "PNG", outputType = gdal.GDT_UInt16)
    png_path = results_path+"merged.png"

    img = Image.open(png_path)
    img = np.asarray(img)
    patches = patchify(img, (244, 244, 3), step=244)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0]
            patch = Image.fromarray(patch)
            num = i * patches.shape[1] + j
            patch.save(f"{patches_folder}/patch_{num}.png")

    return patches_folder, patches



### main
block_path = "/Users/willbyrne/Documents/Code/GLAD/unet_cloud/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/"
block_results_folder = "unet_results/HLS.S30.T19NHA.2021001T144731.v2.0/"


if not os.path.exists('unet_results'):
    os.mkdir('unet_results')

block_dict = get_bands(block_path)

block_results_folder = make_image(block_dict, ["B8A", 'B11', 'B12'])

patches_folder, patches = patchify_image(block_results_folder)

mod_app.model_on_patches(256, 256, block_results_folder)

mod_app.reconstruct_patch_results(block_results_folder)

































# rgb_im = image.convert('RGB')

# image = np.asarray(rgb_im)


# patches = patchify(image, (244, 244, 3), step=244)
# print(patches.shape)  # (6, 10, 1, 512, 512, 3)
# print(type(patches))






# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         patch = patches[i, j, 0]
#         patch = Image.fromarray(patch)
#         num = i * patches.shape[1] + j
#         patch.save(f"/Users/willbyrne/Documents/CODE/GLAD/cloud_mask/data/blocks/HLS.S30.T19NHA.2021001T144731/fmask_fishnet/patch_{num}.png")

# for i in range(0,31):
#     img_num = rand.randrange(1,224,1)
#     shutil.copy(f"./training/fishnet/patch_{img_num}.png", f"training/images/patch_{img_num}.png")





# reconstructed_image = unpatchify(patches, (3660, 3660, 3))
# reconstructed_image = Image.fromarray(reconstructed_image)
# reconstructed_image.save("output.jpg")

# for i in range(1,3660):
# 	if 3660 % i == 0:
# 		print(i)
# 		if i > 200:
# 			break