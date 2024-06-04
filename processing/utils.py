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

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv.threshold(image, 127, 255, 0)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def perform_processing(image: np.ndarray) -> str:

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
    # Remove small contours
    contours = [contour for contour in contours if cv.contourArea(contour) > 1000]
    # Remove contours that are too wide
    contours = [contour for contour in contours if cv.boundingRect(contour)[2] < 0.2 * resized_image.shape[1]]
    # Remove contours that are wider than they are tall
    contours = [contour for contour in contours if cv.boundingRect(contour)[2] < 1.2 * cv.boundingRect(contour)[3]]
    blue_contour = max(contours, key=cv.contourArea)

    # Draw contours
    cv.drawContours(resized_image, contours, -1, (0, 255, 0), 2)

    # Add bilateral filter to the image
    bw_image = cv.bilateralFilter(bw_image, 5, 50, 50)
    cv.imshow('bw_image', bw_image)

    # Add canny edge detection
    edges = cv.Canny(bw_image, 30, 150)
    edges = cv.dilate(edges, np.ones((3,3)), iterations=1)

    cv.imshow('edges', edges)

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
    license_plate_v = license_plate_hsv[:, :, 2]

    license_plate_v = cv.bilateralFilter(license_plate_v, 5, 50, 50)
    license_plate_v = cv.Canny(license_plate_v, 30, 150)
    contours, _ = cv.findContours(license_plate_v, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a list of recognized contours
    recognized_contours = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w < 0.5 * license_plate_v.shape[1] and w > 10 and h > 0.5 * license_plate_v.shape[0]:
            cv.drawContours(license_plate, [contour], -1, (255, 0, 0), 2)
            # Draw the bounding rectangle
            cv.rectangle(license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Skeletonize the image
            skeleton = skeletonize(license_plate_v[y:y+h, x:x+w])
            cv.imshow('skeleton', skeleton)
            cv.imshow("licence plate", license_plate)

            #TODO - recognize the characters from the license plate and add them to the recognized_contours list
            # recognized_contours.append(character)
            # create a string from the recognized_contours list
            # plate = ''.join(recognized_contours)

            # key = cv.waitKey(0)
            # if key == ord('b'):
            #     break

    cv.imshow('license_plate_v', license_plate_v)
    cv.imshow("licence plate", license_plate)


    while True:
        #cv.imshow('mask', mask)
        cv.imshow('image', resized_image)
        # Comment the lines below when not testing manually
        key = cv.waitKey(0)
        if key == ord('a'):
            break

    print(f'image.shape: {image.shape}')
    #TODO: add image processing here
    # return plate

    return 'PO12345'