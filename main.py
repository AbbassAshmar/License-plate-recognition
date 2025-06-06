import argparse
import json
from pathlib import Path

import cv2

from processing.utils import perform_processing
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    results_file = Path(args.results_file)

    images_paths = sorted([image_path for image_path in images_dir.iterdir() if image_path.name.endswith('.jpg')])

    # Load the KNN model
    #knn = cv2.ml.KNearest_load("./processing/knn_model6.xml")
    knn = cv2.ml.KNearest_load("./knn_model5.xml")
    with open("./random_forest_model.pkl", "rb") as file:
        random_forest = pickle.load(file)

    results = {}
    for image_path in images_paths:
        print(f'\n\n\nProcessing image: {image_path}\n\n\n')
        image = cv2.imread(str(image_path))
   
        if image is None:
            print(f'Error loading image {image_path}')
            continue

        results[image_path.name] = perform_processing(image, knn, random_forest)


    with results_file.open('w') as output_file:
        json.dump(results, output_file, indent=4)


if __name__ == '__main__':
    main()

