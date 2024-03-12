# Yoga Pose Detection with Mask R-CNN

## Introduction:

Object Detection Challenges:
Object detection algorithms, such as Mask R-CNN, have made significant strides in computer vision. However, assessing their adaptability to novel objects absent from standard datasets remains a pertinent exploration.

Yoga Pose Recognition:
This project delves into unsupervised object detection by examining the Mask R-CNN algorithm's ability to identify custom objects, particularly yoga poses like downdog. Evaluating its performance on such distinct items contributes to its broader applicability.

## Annotation:

- Dataset Creation:
Annotated a custom dataset of 100 images of downdog poses using the VGG annotator tool. Precisely outlined pose contours with polygon shapes for accurate delineation.

## Mask R-CNN Methodology:

- Model Training:
Fine-tuned the Mask R-CNN model architecture to recognize downdog poses. Leveraged pre-trained COCO weights, excluding certain layers for alignment with the specific task.

- Model Configuration:
Defined a custom configuration class (yogaConfig) to set essential parameters for training, such as the number of classes and GPU utilization. Utilized ResNet-101 architecture as the backbone.

## Prediction Methodology:

- Inference Setup:
Prepared the model for inference using the custom configuration class (SimpleConfig). Loaded trained weights and performed forward pass for pose detection in new images.

- Visualization:
Visualized detected downdog poses on input images to assess the model's performance.

## Results:

- Performance Evaluation:
Rigorously evaluated the Mask R-CNN model's performance on original and synthetic images. Key metrics include acceptance rate, false positive rate (FPR), and false negative rate (FNR).

- Insights:
Highlighted successful downdog pose detections and potential misclassifications, showcasing the model's nuanced performance.

## Conclusion:

- Project Outcome:
Successfully applied Mask R-CNN algorithm for unsupervised object detection, achieving a commendable acceptance rate of 90.62% on original images. Demonstrated the algorithm's adaptability and effectiveness beyond its initial training data.

- Contribution:
Provides practical insights into the algorithm's potential for real-world applications, contributing to the discourse on object detection algorithms.
## How to use:
Set Up Root Directory:

Establish a root directory (e.g., Object Detection).
Copy Necessary Files:

Within the root directory, duplicate the 'mrcnn' directory.
Download Pre-trained Weights:

Retrieve the pre-trained weights and place them in the root directory.
Access the weights via this link: Pre-trained Weights.
Create Object Detection Script:

Craft a script dedicated to object detection.
Save this script within the root directory. An exemplary script can be found at: samples/mask-rcnn-prediction.py.
Refer to the subsequent section for the script's code.
Execute the Script:

Run the script to initiate the object detection process.

## License:

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, provide feedback, or use this project for your own research or applications.
