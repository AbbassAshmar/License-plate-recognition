# License plate recognition

Goal of the project is to recognize license plates on images without using deep learning methods. Project was prepared for course "Vision systems". Final accuracy on test set was around 80-85%.

![Result example](/media/license_plate_recognition.png)

## Table of contents
<!-- TOC -->
* [License plate recognition](#license-plate-recognition)
  * [Table of contents](#table-of-contents)
  * [1. Requirements](#1-requirements)
  * [2. Data](#2-data)
  * [3. Structure of the project](#3-structure-of-the-project)
  * [4. How to run the project](#4-how-to-run-the-project)
  * [5. About the algorithm](#5-about-the-algorithm)
  * [6. Training the classifiers](#6-training-the-classifiers)
  * [7. Results](#7-results)
<!-- TOC -->

## 1. Requirements

The project was written for Python 3.11. Requirements are listed in `requirements.txt` file. To install them run:

```
pip install -r requirements.txt
```

## 2. Data

The training set containing 26 images of license plates was shared by the course lecturer. There were some rules for the data:
- all license plates were from Poland,
- all license plates were white (no EV or temporary plates),
- license plates were rotated by maximum 45 degrees,
- license plates width was at least 1/3 of the image width.

## 3. Structure of the project

The project is divided into several files:
- [main script](main.py)
- [image processing script](/processing/utils.py)
- [pictures of separate letters](/pictures)
- [script to calculate the result](/pictures/count_points.py)
- [script to create and train KNN and Random Forest Classifiers](/pictures/knn_model_training.py)

## 4. How to run the project
Before first run you need to train the classifiers. To do this, run the script `knn_model_training.py` in the `pictures` directory. It will create and save the models in the same directory.

Once you have the model trained, to run the project you need a training set of images in `.jpg` format. Copy path to the directory with images and path where do you want to save the results and run it in the terminal:

```
python main.py /path/to/training_dataset /path/to/results/results.json
```

## 5. About the algorithm

Image processing inside the [utils.py](/processing/utils.py) script can be divided into the following steps:
1. Resize the image to 1920x1080 and create a greyscale and hsv version of it.
2. Create a mask for blue color and apply it to the hsv image. After doing so find contours and apply some rules (like for example height to width ratio) to find the rectangle that is always on the left side of the license plate.
3. Filter the greyscale image with bilateral filter and use Canny detector to find edges. Dilate the edges and find contours. Create a bounding rectangle for each contour and check it's position comparing to the blue rectangle and remove contours with small area. After that use convexHull to get the contour as polygon.
4. Sort hulls by the area and choose the smallest one. After that approximate the polygon with 4 points (there is a function that is increasing the difference between the points and the original polygon to finally find only 4 points).
5. Use the 4 points to warp the image. Now the license plate is straight.
6. Take the v channel of hsv image of the license plate and filter it, use Canny detenctor and find contours.
7. Iterate through the found contours and remove all that are too wide or not high enough. After that create a bounding rectangle for each contour. Scale the rectangle to be height of 90 pixels and add padding with black pixels to both sides so the width is 66 pixels. Add the new image to a dictionary with the x coordinate being a key.
8. Remove images that are inside another one (happened with the inside and outside of 0).
9. Iterate through the dictionary (sorted by x coordinate) and predict the letter with the KNN and Random Forest classifiers. Save the predictions to the list.
10. If length of the list is longer than 8, remove the elements with indices >=8. Check if the first or second element is a number, if so than change it to a letter that has similar shape.
11. Save the results to the json file.

## 6. Training the classifiers

Because the training set wasn't big, the classifiers were trained on pictures that were randomly deformed using the `elasticdeform` library. In addition there was a function to squeeze the pictures to add even more differences. By doing so, from one picture there were 50 new images that were used for training and 10 new images for validation. The accuracy on validation set for both KNN and Random Forest classifiers was around 99%.

![Create the dataset](/media/classifier_data_creation.gif)

## 7. Results

![Result](/media/license_plate_example.png)