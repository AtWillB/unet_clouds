# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Input, Conv2D
# from keras.models import Model
# from keras.utils import normalize
# # import segmentation_models as sm
# from sklearn.model_selection import train_test_split
import augment_trainingset



#path sorting
import glob
from osgeo import gdal
from pathlib import Path
import re
import json

#math
import numpy as np
import matplotlib.pyplot as plt
import math

#regex pattern I copied to sort filepaths in ascending order
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])


def display_results(results_path):
    model_path_no_ext = results_path.split(".")[0]
    results_path =  model_path_no_ext+".json"
    
    
    with open(results_path) as json_file:
        results = json.load(json_file)

    print("Which model is this? -",results_path.split("/")[-1])
    type = results_path.split("/")[-1].split("_")[0]

    iou_score = results['iou_score']
    val_iou_score = results['val_iou_score']
    loss = results['loss']
    val_loss = results['val_loss']

    epochs = range(1, len(iou_score) + 1)

    plt.plot(epochs, iou_score, 'bo', label='Training acc')
    plt.plot(epochs, val_iou_score, 'b', label='Validation acc')
    plt.plot(val_iou_score.index(max(val_iou_score)), max(val_iou_score), 'ro', label='Best val acc')
    plt.title(f'{type} Spoke Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.yscale("log")
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.plot(val_iou_score.index(max(val_iou_score)), val_loss[val_iou_score.index(max(val_iou_score))], 'ro', label='Best val acc')
    plt.plot(val_loss.index(max(val_loss)), val_loss, 'bo', label='Best val acc')
    plt.title(f'{type} Spoke Training and validation loss')
    plt.legend()

    plt.show()
    print("Last Train IOU Score: ",results['iou_score'][-1])
    print("Last Train Loss Score: ", results['loss'][-1])
    print("Last Validation IOU Score: ", results['val_iou_score'][-1])
    print("Last Validation Loss Score: ", results['val_loss'][-1])
    json_file.close()



def data_gather(X, Y, image_type="cloud_training_images", mask_type="cloud_training_masks", training_path = "../datasets/", aug_flag = 0, aug_num = 0):
    print()

    for img_path in sorted(glob.glob(training_path+image_type+"/*.tif"), key=get_order):
        img_chip = gdal.Open(img_path)
        img_chip = img_chip.ReadAsArray()
        X.append(img_chip)

    for mask_path in sorted(glob.glob(training_path+mask_type+"/*.tif"), key=get_order):
        mask_chip = gdal.Open(mask_path)
        mask_chip = mask_chip.ReadAsArray()
        Y.append(mask_chip)

    if aug_flag == 1:
        X, Y = augment_trainingset.augment_semantic_set(X, Y, aug_num = aug_num)
    
    print(len(X), len(Y))    

    
    
    return X, Y



def fit_model(x_train, y_train, model, model_path,batch_size = 10,epochs = 300, validation_split = .15 ):
    #print(model.summary())
    
    # fit model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_iou_score',
    mode='max',
    save_best_only=True, 
    verbose = True)
    
    history = model.fit(
       x = x_train,
       y = y_train,
       batch_size = batch_size,
       epochs = epochs,
       verbose = 1,
       validation_split = validation_split,
       callbacks = [model_checkpoint_callback]
    )

    return history



def define_model(SIZE_Y, SIZE_X, backbone = "resnet34"):   

    model = sm.Unet(backbone_name="resnet34", encoder_weights = None, input_shape=(10, SIZE_Y,SIZE_X), classes = 1)
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = [sm.metrics.IOUScore()], )
    print(model.summary())
    return model



def save_model_history(model_path, model, history, results):

    model_path_no_ext = model_path.split(".")[0]
    print(f"Which model is this:  {model_path_no_ext}")

    dump_dict = history.history
    dump_dict['eval_results'] = results

    with open(f"{model_path_no_ext}.json", 'w') as f:
        json.dump(dump_dict, f)
    f.close()



def model_testing(model, testing_folder, num_of_images):
    remaining_dataset = sorted(glob.glob(f"../datasets/{testing_folder}*.tif"), key=get_order)
    remaining_test = []
    filenames = []

    print(f"The {testing_folder} training set is made of {len(remaining_dataset)} images")

    for img_path in remaining_dataset:
        filenames.append(img_path.split("/")[-1])
        img_chip = rasterio.open(img_path)
        img_chip = img_chip.read()
        remaining_test.append(img_chip)
    
    remaining_test = np.array(remaining_test)

    # num_of_images problematic here
    for filename, img in zip(filenames[:num_of_images], remaining_test[:num_of_images]):
        print(filename, filenames.index(f"{filename}"))                                         
        plt.imshow(img, cmap="gray")
        plt.show()

        img = img.reshape((1, 256, 256))    
        prediction = model.predict(img)
        prediction = prediction.reshape((256, 256))

        plt.imshow(prediction, cmap='gray')
        plt.show()
    
    plt.close()
    return