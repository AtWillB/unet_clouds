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

from osgeo import gdal




# Grab the bands from the specified block, and create a dictionary that stores the path to 
# all of the blocks bands, and the blocks name.
# Return the dictionary
def get_bands(block_path):
    block_dict = {}
    block_name = block_path.split("/")[-2]
    hls_type = block_name.split(".")[1]

    block_dict.update({"block_name": block_name})
    block_dict.update({"hls_type": hls_type})

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


    os.system('gdal_merge.py -o '+results_path+"merged.tiff " + image_band_list[0]+" "+image_band_list[1]+" "+image_band_list[2]+' -separate')
    print("\n")

    return results_path


# Create a png of the geotiff(program was trained on pngs, but conversion may be unessesary)
# Devide the merged image into patches, and create a folder for those divided patches. 
# Returns the path to the patches folder
def patchify_image(results_path):
    patches_folder = results_path+"patches"
    if not os.path.exists(patches_folder):
        os.mkdir(patches_folder)

    png_path = results_path+"merged.png"
    os.system(f"gdal_translate -of PNG -scale {results_path}"+f"merged.tiff {png_path}")
    
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
block_path = "/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/"
block_results_folder = "unet_results/HLS.S30.T19NHA.2021001T144731.v2.0/"
block_dict = {'block_name': 'HLS.S30.T19NHA.2021001T144731.v2.0', 
'B09': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B09.tif', 
'B08': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B08.tif', 
'B8A': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B8A.tif', 
'B01': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B01.tif', 
'B03': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B03.tif', 
'B02': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B02.tif', 
'B06': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B06.tif', 
'B12': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B12.tif', 
'B07': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B07.tif', 
'B11': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B11.tif', 
'B05': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B05.tif', 
'B04': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B04.tif', 
'B10': '/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/s30/HLS.S30.T19NHA.2021001T144731.v2.0/HLS.S30.T19NHA.2021001T144731.v2.0.B10.tif'}



for block_path in glob.glob("/Users/willbyrne/Documents/work/code/glad/unet_clouds/data/hls_2021_data/**/*.v2.0/", recursive=True):
    if not os.path.exists('unet_results'):
        os.mkdir('unet_results')

    block_dict = get_bands(block_path)
    if block_dict['hls_type'] == "S30":
        block_results_folder = make_image(block_dict, ["B8A", 'B11', 'B12'])
    elif block_dict["hls_type"] == "L30":
        block_results_folder = make_image(block_dict, ["B05", 'B06', 'B07'])

    patches_folder, patches = patchify_image(block_results_folder)

    mod_app.model_on_patches(256, 256, block_results_folder)
    mod_app.reconstruct_patch_results(block_results_folder)
    print(f"\n*****\nCOMPLETED {block_dict['block_name']}\n*****\n")