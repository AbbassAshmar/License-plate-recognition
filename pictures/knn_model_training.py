import cv2 as cv
import numpy as np
import os
import elasticdeform
from sklearn.ensemble import RandomForestClassifier
import pickle

def filter_image(image):
    deformed_image_filtered = cv.medianBlur(image, 5)
    _, deformed_image_filtered_1 = cv.threshold(deformed_image_filtered, 20, 255, cv.THRESH_BINARY)
    return deformed_image_filtered_1

def squeeze_image(image):
    # Get the height and width of the image
    height, width = image.shape

    # Define the source points
    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    squeeze_factor = (np.random.rand() - 0.5) * 0.06

    # Define the destination points
    dst_pts = np.float32([[width * squeeze_factor, 0], [width * (1-squeeze_factor), 0], [width * squeeze_factor, height], [width * (1-squeeze_factor), height]])

    # Get the transformation matrix
    matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the transformation
    squeezed_image = cv.warpPerspective(image, matrix, (width, height))

    return squeezed_image

def main():
    #letters_and_numbers = []
    letters_and_numbers = os.listdir(".")

    # Lists to store the images and the labels
    images = []
    labels = []

    # Lists to store the images and the labels for validation set
    images_validation = []
    labels_validation = []

    for file in letters_and_numbers:
        if file.endswith(".png"):
            if file[1] == "_":
                image = cv.imread(file, cv.IMREAD_GRAYSCALE)
            else:
                image = cv.imread(file, cv.IMREAD_GRAYSCALE)
                # Resize the image - 80 is height of letters on the license plate and 173 is the height of the letter on image
                image = cv.resize(image, (0, 0), fx=80/173, fy=80/173)
                # change colors (black to white, white to black)
                image = cv.bitwise_not(image)
            print(file)
            for i in range(0, 50):
                # Random rotation (-10; 10), zoom (0.9; 1.1) and number of points (5; 15)
                rotation = (np.random.rand() - 0.5) * 20
                zoom = np.random.rand() * 0.2 + 0.9
                number_of_points = np.random.randint(5, 15)
                # Deform the image and filter it
                deformed_image = elasticdeform.deform_random_grid(image, sigma=0.75, points=number_of_points, rotate=rotation, zoom=zoom)
                deformed_image = squeeze_image(deformed_image)
                deformed_image_filtered = filter_image(deformed_image)
                # Append the flattened image and the label to the lists
                images.append(deformed_image_filtered.flatten())
                labels.append(file[0])

            # Create validation set
            for i in range(0, 10):
                # Random rotation (-10; 10), zoom (0.9; 1.1) and number of points (5; 15)
                rotation = (np.random.rand() - 0.5) * 20
                zoom = np.random.rand() * 0.2 + 0.9
                number_of_points = np.random.randint(5, 15)
                # Deform the image and filter it
                deformed_image = elasticdeform.deform_random_grid(image, sigma=0.75, points=number_of_points, rotate=rotation, zoom=zoom)
                deformed_image = squeeze_image(deformed_image)
                deformed_image_filtered = filter_image(deformed_image)
                # Append the flattened image and the label to the lists
                images_validation.append(deformed_image_filtered.flatten())
                labels_validation.append(file[0])

    # Convert the lists to numpy arrays
    images = np.array(images)
    images = images.astype(np.float32)
    labels = np.array(labels)
    images_validation = np.array(images_validation)
    images_validation = images_validation.astype(np.float32)
    labels_validation = np.array(labels_validation)

    unique_chars = np.unique(labels)
    print(unique_chars)
    char_to_int = {char: i for i, char in enumerate(unique_chars)}

    # Convert the labels to integers
    labels = np.array([char_to_int[char] for char in labels])
    labels_validation = np.array([char_to_int[char] for char in labels_validation])

    # Create a KNN model
    knn = cv.ml.KNearest_create()
    # Train the model
    knn.train(images, cv.ml.ROW_SAMPLE, labels)
    # Save the model
    knn.save("knn_model6.xml")

    # Load the model
    knn = cv.ml.KNearest_load("knn_model6.xml")

    # Write the accuracy of the model to the console
    print(images_validation.shape)
    _, result, _, _ = knn.findNearest(images_validation, 5)
    result = result.flatten()
    accuracy = np.mean(result == labels_validation)
    # for i in range(len(result)):
    #     print(f"Predicted: {unique_chars[int(result[i])]}, Actual: {unique_chars[labels_validation[i]]}")
    print(f"Accuracy on validation set using k-Nearest Neighbors: {accuracy}")

    # Create a random forest model
    random_forest = RandomForestClassifier(n_estimators=100)
    # Train the model
    random_forest.fit(images, labels)
    # Save the model
    with open("random_forest_model.pkl", "wb") as file:
        pickle.dump(random_forest, file)

    # Load the model
    with open("random_forest_model.pkl", "rb") as file:
        random_forest = pickle.load(file)

    # Write the accuracy of the model to the console
    accuracy = random_forest.score(images_validation, labels_validation)
    print(f"Accuracy on validation set using Random Forest: {accuracy}")


if __name__ == "__main__":
    main()
