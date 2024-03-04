from PIL import Image
import cv2
import numpy as np

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class CropLungsTransform:
    def __init__(self, model, avoid_postprocess=False, task='mask'):
        self.model = model
        self.avoid_postprocess = avoid_postprocess
        self.task = task

    def __call__(self, image):
        # image = transforms.Grayscale(num_output_channels=1)(image)
        if self.task == 'centroid':
            return self.postprocess_mask(im=image)
        mask = self.postprocess_mask(im=image)
        cropped = self.crop(image, mask)
        return np.dstack([cropped, cropped, cropped])
        # return cropped

    def crop(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.resize((224, 224))
        mask = mask.resize((224, 224))
        image = transforms.Grayscale(num_output_channels=1)(image)

        image = np.array(image)
        mask = np.array(mask)
        
        cropped = cv2.bitwise_and(image, mask)
        cropped = Image.fromarray(cropped)
        return cropped
        
    def postprocess_mask(self, im):

        model=self.model
        return_original=self.avoid_postprocess

        model = model.cpu()
        IMG_SIZE = 224

        class ContrastTransform:
            def __init__(self, factor):
                self.factor = factor

            def __call__(self, x):
                return TF.adjust_contrast(x, self.factor)

        transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    ContrastTransform(1.8),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToPILImage(),
                    transforms.ToTensor()
        ])

        im = transform(im)
        im = im.unsqueeze(0)
        res = model(im)[:,0,:,:]
        res = torch.round(res)

        res = transforms.ToPILImage()(res)

        if return_original:
            return res

        # ConnectedComponentLabelling
        input = np.expand_dims(np.array(res), -1)

        # Output positions and area sizes are stored in stats[i], mask itself is stored in labels[i]
        output = cv2.connectedComponentsWithStats(
            input, connectivity=8, ltype=cv2.CV_32S)
        (num_labels, labels, stats, centroids) = output

        # Initialise the output shape
        mask = np.zeros(input.shape, dtype="uint8")

        # Get the areas (~usually the two lungs and the two bits)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(num_labels)]
        

        # Find two largest areas: sort descending -> select 2, not including first one which is whole image
        two_largest_areas = np.argsort(areas)[::-1][1:3]

        # Get centroids of lungs
        lung_centroids = centroids[two_largest_areas]

        for component in two_largest_areas:
            componentMask = (labels == component).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

        DILATE_SIZE = 5
        DILATE_ITERATIONS = 3
        ERODE_SIZE = 5
        ERODE_ITERATIONS = 3
        FINAL_DILATE_SIZE = 2

        dilate_kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
        erode_kernel = np.ones((ERODE_SIZE, ERODE_ITERATIONS), np.uint8)
        final_dilate_kernel = np.ones((FINAL_DILATE_SIZE, FINAL_DILATE_SIZE), np.uint8)

        eroded = cv2.erode(mask, erode_kernel, iterations=ERODE_ITERATIONS)
        dilated = cv2.dilate(eroded, dilate_kernel, iterations=DILATE_ITERATIONS)
        final_mask = cv2.dilate(dilated, final_dilate_kernel, iterations=1)

        final_mask = Image.fromarray(final_mask)

        if self.task == 'mask':
            return final_mask
        elif self.task == 'centroid':
            return lung_centroids
        raise Exception('Please specify task: "mask" or "centroid"')