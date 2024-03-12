import os
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros, asarray
from keras.callbacks import Callback

import mrcnn.utils
import mrcnn.config
import mrcnn.model
from sklearn.model_selection import train_test_split

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.class_losses = []
        self.rpn_losses = []
        self.box_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.class_losses.append(logs.get('class_loss'))
        self.rpn_losses.append(logs.get('rpn_class_loss'))
        self.box_losses.append(logs.get('mrcnn_bbox_loss'))
        # Add validation losses for class, rpn, and box
        self.class_val_losses.append(logs.get('val_class_loss'))
        self.rpn_val_losses.append(logs.get('val_rpn_class_loss'))
        self.box_val_losses.append(logs.get('val_mrcnn_bbox_loss'))

class yogaDataset(mrcnn.utils.Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "yoga")

        if is_train:
            images_dir=dataset_dir+'/train/images/'
            annotations_dir=dataset_dir+'/train/annots/'
            
        else:
            images_dir=dataset_dir+'/val/images/'
            annotations_dir=dataset_dir+'/val/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.json'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('yoga'))
        return masks, asarray(class_ids, dtype='int32')

    def extract_boxes(self, filename):
        with open(filename) as f:
            data = json.load(f)

        boxes = list()
        for key in data.keys():
            regions = data[key]['regions']
            for _, region in regions.items():
                shape_attributes = region['shape_attributes']
                all_points_x = shape_attributes['all_points_x']
                all_points_y = shape_attributes['all_points_y']
                xmin = min(all_points_x)
                ymin = min(all_points_y)
                xmax = max(all_points_x)
                ymax = max(all_points_y)
                coors = [xmin, ymin, xmax, ymax]
                boxes.append(coors)

        width = 512  # replace with your image width
        height = 512  # replace with your image height
        return boxes, width, height

class yogaConfig(mrcnn.config.Config):
    NAME = "yoga_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    STEPS_PER_EPOCH = 100

# Train dataset
train_dataset = yogaDataset()
train_dataset.load_dataset(dataset_dir='E:/M2/Machine learning/Object Detection/yoga', is_train=True)
train_dataset.prepare()

# Validation dataset
validation_dataset = yogaDataset()
validation_dataset.load_dataset(dataset_dir='E:/M2/Machine learning/Object Detection/yoga', is_train=False)
validation_dataset.prepare()

# Model Configuration
yoga_config = yogaConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=yoga_config)

model.load_weights(filepath='E:/M2/Machine learning/Object Detection/mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# Train the model and retrieve the training history using the custom callback
history_callback = LossHistory()
model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=yoga_config.LEARNING_RATE, 
            epochs=40,  # Choose the number of epochs you want
            layers='heads',
            custom_callbacks=[history_callback])

# Plot the training and validation loss
plt.figure(figsize=(8, 6))

plt.plot(np.arange(1, len(history_callback.losses) + 1), history_callback.losses, label='Train Loss')
plt.plot(np.arange(1, len(history_callback.val_losses) + 1), history_callback.val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'Train', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Plot the class loss
plt.figure(figsize=(8, 6))

plt.plot(np.arange(1, len(history_callback.class_losses) + 1), history_callback.class_losses, label='Class Loss')
plt.plot(np.arange(1, len(history_callback.class_val_losses) + 1), history_callback.class_val_losses, label='Val Class Loss')
plt.title('Training and Validation Class Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'Class', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Plot the RPN loss
plt.figure(figsize=(8, 6))

plt.plot(np.arange(1, len(history_callback.rpn_losses) + 1), history_callback.rpn_losses, label='RPN Loss')
plt.plot(np.arange(1, len(history_callback.rpn_val_losses) + 1), history_callback.rpn_val_losses, label='Val RPN Loss')
plt.title('Training and Validation RPN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'RPN', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Plot the box loss
plt.figure(figsize=(8, 6))

plt.plot(np.arange(1, len(history_callback.box_losses) + 1), history_callback.box_losses, label='Box Loss')
plt.plot(np.arange(1, len(history_callback.box_val_losses) + 1), history_callback.box_val_losses, label='Val Box Loss')
plt.title('Training and Validation Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.text(0.5, 0.5, 'Box', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# Save the trained model
model_path = 'downdog_mask_rcnn_coco_datasplit40.h5'
model.keras_model.save_weights(model_path)
