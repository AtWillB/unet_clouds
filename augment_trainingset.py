import albumentations as alb
from skimage import io
from os.path import exists
import numpy as np




#Albumentation function
# list of potential augmentations

# 1. InvertImg() - Invert the input image via the following formula per pixel: x = |x-255|
# 2. JpegCompression() - Decreases image quality by Jpeg compression of an image
# 3. MultiplicativeNoisse() - Multiply image to random number or array of numbers
# 5. PixelDropout() - Set pixels to 0 with some probability.
# 6. RandomBrightness() - Randomly change brightness of the input image.
# 7. RandomBrightnessContrast() - Randomly change brightness and contrast of the input image.
# 8. RandomContrast() - Randomly change contrast of the input image
# 9. RandomGamma() - Randomly change gamma of the input image
# 10. RandomGridShuffle() - Divide the image into a grid of (x, y) size. Then randomly shuffle this grids cells. 
# 12. RandomToneCurve() - Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.
# 13. RingingOvershoot() - Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.
# 14. Sharpen() - Sharpen the input image and overlays the result with the original image.
# 15. Solarize() - Invert all pixel values above a threshold.
# 16. AdvancedBlur() - Blur the input image using a Generalized Normal filter with a randomly selected parameters. This transform also adds multiplicative noise to generated kernel before convolution.
# 17. Blur() - Blur the input image using a random-sized kernel.
# 18. GaussianBlur() - Blur the input image using a Gaussian filter with a random kernel size.
# 19. MedianBlur() - Blur the input image using a median filter with a random aperture linear size.
# 20. MotionBlur() - Apply motion blur to the input image using a random-sized kernel.
# 21. Flip() - Flip the input either horizontally, vertically or both horizontally and vertically.
# 22. HorizontalFlip() - Flip the input horizontally around the y-axis.
# 23. ShiftScaleRotate() - Randomly apply affine transforms: translate, scale and rotate the input.
# 24. Transpose() - Transpose the input by swapping rows and columns.
# 25. VerticalFlip() - Flip the input vertically around the x-axis.
# 26. CLAHE() - Apply Contrast Limited Adaptive Histogram Equalization to the input image. 
# 27. RandomScale() - Randomly resize the input. Output image size is different from the input image size.



# Selection
#1 VerticalFlip()
#2 GaussianBlur()
#3 RandomToneCurve()
#4 RandomBrightnessContrast()
#5 PixelDropout()
#6 CLAHE()
#7 Sharpen()




aug = alb.Compose([
    alb.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit_y = 0, p=1),
    alb.RandomBrightnessContrast (brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=True, p=1)], 
    additional_targets={'image' : 'mask'}
)

def augment_semantic_set(X, Y, aug_num = 2):
    folder_path = "data/training/dark/augmentation/"

    for image, mask in zip(X,Y):
            for i in range(0, aug_num):
                augs = aug(image = image, mask = mask)
                X = np.append(X, [augs['image']], axis=0)
                Y = np.append(Y, [augs['mask']], axis=0)

    return X,Y
    



	