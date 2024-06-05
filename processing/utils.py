import numpy as np
import cv2 as cv

def approximate_to_4_points(contour):
    epsilon = 0.001 * cv.arcLength(contour, True)
    while True:
        approx = cv.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
        epsilon += 0.001 * cv.arcLength(contour, True)


# Function to sort the points
def sort_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    # Sum of the points (top-left has the smallest sum, bottom-right has the largest sum)
    pts = np.array(pts)
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of the points (top-right has the smallest difference, bottom-left has the largest difference)
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perform_processing(image: np.ndarray, knn, random_forest) -> str:

    resized_image = cv.resize(image, (1920, 1080))
    clear_resized_image = resized_image.copy()
    bw_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    hsv_image = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)

    # checking color on the license plate
    # def click_event(event, x, y, flags, param):
    #     if event == cv.EVENT_LBUTTONDOWN:
    #         print(hsv_image[y, x])
    # while True:
    #     cv.imshow('image', hsv_image)
    #     cv.setMouseCallback('image', click_event)
    #     key = cv.waitKey(0)
    #     if key == ord('a'):
    #         break

    # From the above code (I've checked 322 pixels in total across 26 pictures) the results are:
    # for the blue color:
    # H - 102-115, S - 127-255, V - 90-227

    # Create a mask for the blue color
    lower_blue = np.array([102, 127, 90])
    upper_blue = np.array([113, 255, 227])
    mask = cv.inRange(hsv_image, lower_blue, upper_blue)

    # Closing the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_copy = contours
    # Remove small contours
    contours = [contour for contour in contours if cv.contourArea(contour) > 1000]
    # Remove contours that are too wide
    contours = [contour for contour in contours if cv.boundingRect(contour)[2] < 0.2 * resized_image.shape[1]]
    # Remove contours that are wider than they are tall
    contours = [contour for contour in contours if cv.boundingRect(contour)[2] < 1.2 * cv.boundingRect(contour)[3]]
    if len(contours) == 0:
        contours = contours_copy
    blue_contour = max(contours, key=cv.contourArea)

    # Draw contours
    cv.drawContours(resized_image, contours, -1, (0, 255, 0), 2)

    # Add bilateral filter to the image
    bw_image = cv.bilateralFilter(bw_image, 5, 50, 50)
    # cv.imshow('bw_image', bw_image)

    # Add canny edge detection
    edges = cv.Canny(bw_image, 30, 150)
    edges = cv.dilate(edges, np.ones((3,3)), iterations=1)

    # cv.imshow('edges', edges)
    # cv.waitKey(0)

    # Find contours of the edges
    edge_contours, _ = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Find the contours that are next to the blue contour
    hulls = []
    xb, yb, wb, hb = cv.boundingRect(blue_contour)
    for contour in edge_contours:
        # Skip small contours
        if cv.contourArea(contour) < 10e4:
            continue
        x, y, w, h = cv.boundingRect(contour)
        # Skip contours that are not placed correctly
        if x < xb or x > xb + wb or y > yb + hb or y + h < yb:
            continue
        # Get contour as a polygon - game changer
        hull = cv.convexHull(contour)
        hulls.append(hull)

    if len(hulls) == 0:
        print('No hulls found')
        return 'PO12345'

    # Sort hulls to get the smallest one - the one that is around the license plate - and extract it corners
    hulls.sort(key=lambda x: cv.contourArea(x))
    approx_4_points = approximate_to_4_points(hulls[0])
    original_pts = [(approx_4_points[0][0][0], approx_4_points[0][0][1]),
                    (approx_4_points[1][0][0], approx_4_points[1][0][1]),
                    (approx_4_points[2][0][0], approx_4_points[2][0][1]),
                    (approx_4_points[3][0][0], approx_4_points[3][0][1])]
    sorted_original_pts = sort_points(original_pts)
    # Create a list of points for perspective transformation
    new_pts = np.float32([(0, 0), (463, 0), (463, 99), (0, 99)])
    matrix = cv.getPerspectiveTransform(sorted_original_pts, new_pts)
    # Warp the image
    license_plate = cv.warpPerspective(resized_image, matrix, (464, 100))

    cv.drawContours(resized_image, [approx_4_points], -1, (0, 0, 255), 2)

    license_plate_hsv = cv.cvtColor(license_plate, cv.COLOR_BGR2HSV)
    license_plate_bw = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)
    license_plate_v = license_plate_hsv[:, :, 2]

    license_plate_v = cv.bilateralFilter(license_plate_v, 5, 50, 50)
    license_plate_v = cv.Canny(license_plate_v, 30, 150)
    contours, _ = cv.findContours(license_plate_v, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a list of recognized contours
    recognized_contours = {}
    letters = 0
    contours_width = {}

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w < 0.5 * license_plate_v.shape[1] and w > 10 and h > 0.5 * license_plate_v.shape[0]:
            cv.drawContours(license_plate, [contour], -1, (255, 0, 0), 2)
            # Draw the bounding rectangle
            cv.rectangle(license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)

            letter_image = license_plate_bw[y:y+h, x:x+w]
            # Get the original width and height of the letter image
            original_height, original_width = letter_image.shape
            letter_image = cv.resize(letter_image, (0, 0), fx=90/original_height, fy=90/original_height)
            mean = np.mean(letter_image)
            _, letter_image = cv.threshold(letter_image, mean-10, 255, cv.THRESH_BINARY)
            letter_image = cv.bitwise_not(letter_image)
            letter_image = cv.morphologyEx(letter_image, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

            # Add padding to the image
            letter_image_height, letter_image_width = letter_image.shape
            padding = 66 - letter_image_width
            if padding < 0:
                letter_image = cv.resize(letter_image, (66, 90))
            else:
                if padding % 2 == 0:
                    letter_image = cv.copyMakeBorder(letter_image, 0, 0, int(padding/2), int(padding/2), cv.BORDER_CONSTANT, value=0)
                else:
                    letter_image = cv.copyMakeBorder(letter_image, 0, 0, int(np.floor(padding/2)), int(np.floor(padding/2))+1, cv.BORDER_CONSTANT, value=0)

            # Add the letter to the recognized_contours list with x coordinate
            # This allows to sort the order of the letters
            recognized_contours[x] = letter_image
            contours_width[x] = w
            #print(letter_image.shape[0], letter_image.shape[1])

            # cv.imshow("licence plate", license_plate)
            letters += 1

            #TODO - recognize the characters from the license plate and add them to the recognized_contours list
            # recognized_contours.append(character)
            # create a string from the recognized_contours list
            # plate = ''.join(recognized_contours)

            # key = cv.waitKey(0)
            # if key == ord('b'):
            #     break

    # Go through each element of the contours and remove contours that are inside each other
    for key in sorted(contours_width.keys()):
        for key2 in sorted(contours_width.keys()):
            if key != key2:
                if key > key2 and key + contours_width[key] < key2 + contours_width[key2]:
                    recognized_contours.pop(key)
                    contours_width.pop(key)

    # Create a list of recognized characters
    recognized_characters = []
    unique_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    char_to_int = {char: i for i, char in enumerate(unique_chars)}

    for key in sorted(recognized_contours.keys()):

        im = recognized_contours[key].flatten()
        im = np.array(im)
        im = im.astype(np.float32)
        im = im.reshape(1, -1)
        print(f'Image shape: {im.shape}')

        # Find the nearest neighbors
        _, result, _, _ = knn.findNearest(im, 9)
        result = result.flatten()
        result_random_forest = random_forest.predict(im)
        #print(f'Result: {result}')
        # Get the character
        character = unique_chars[int(result[0])]
        character_random_forest = unique_chars[int(result_random_forest[0])]

        print(f'Character using KNN: {character}')
        print(f'Character using Random Forest: {character_random_forest}')
        recognized_characters.append(character)


        cv.imshow('recognized_contours', recognized_contours[key])
        # cv.waitKey(0)

    print(f'Letters: {letters}')
    # cv.imshow('license_plate_v', license_plate_v)
    cv.imshow("licence plate", license_plate)
    if len(recognized_characters) > 8:
        recognized_characters = recognized_characters[:8]

    # Replace the characters with the correct ones
    for i in range(2):
        if recognized_characters[i] == '0':
            recognized_characters[i] = 'O'
        if recognized_characters[i] == '1':
            recognized_characters[i] = 'I'
        if recognized_characters[i] == '2':
            recognized_characters[i] = 'Z'
        if recognized_characters[i] == '3':
            recognized_characters[i] = 'E'
        if recognized_characters[i] == '4':
            recognized_characters[i] = 'A'
        if recognized_characters[i] == '5':
            recognized_characters[i] = 'S'
        if recognized_characters[i] == '6':
            recognized_characters[i] = 'G'
        if recognized_characters[i] == '7':
            recognized_characters[i] = 'Z'
        if recognized_characters[i] == '8':
            recognized_characters[i] = 'B'
        if recognized_characters[i] == '9':
            recognized_characters[i] = 'G'

    plate = ''.join(recognized_characters)

    # while True:
    #     #cv.imshow('mask', mask)
    #     cv.imshow('image', resized_image)
    #     # Comment the lines below when not testing manually
    #     # key = cv.waitKey(0)
    #     # if key == ord('a'):
    #     #     break

    print(f'image.shape: {image.shape}')
    #TODO: add image processing here
    # return plate

    return plate