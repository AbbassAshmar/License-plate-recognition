import numpy as np
import cv2 as cv

def calculate_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    points = np.array(pts[0])
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def perform_processing(image: np.ndarray) -> str:

    resized_image = cv.resize(image, (1920, 1080))
    clear_resized_image = resized_image.copy()
    bw_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    hsv_image = cv.cvtColor(resized_image, cv.COLOR_BGR2HSV)
    empty = np.zeros_like(resized_image)

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
    #cv.drawContours(empty, contours, -1, (255, 255, 255), cv.FILLED)


    # Detect edges on bw image
    bw_image = cv.GaussianBlur(bw_image, (5, 5), 0)
    #bw_image = cv.medianBlur(bw_image, 5)

    _, bw_image = cv.threshold(bw_image, 150, 255, cv.THRESH_OTSU)
    bw_image = cv.GaussianBlur(bw_image, (5, 5), 0)
    cv.imshow('bw_image', bw_image)
    edges = cv.Canny(bw_image, 30, 255)
    edges = cv.dilate(edges, None, iterations=1)

    cv.imshow('edges', edges)

    edge_contours, _ = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    rectangular_contours = []
    xb, yb, wb, hb = cv.boundingRect(blue_contour)
    boxes = {}
    for contour in edge_contours:
        if cv.contourArea(contour) < 10e4:
            continue
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(resized_image, [box], 0, (255, 0, 0), 2)
        x, y, w, h = cv.boundingRect(contour)
        if x < xb or x > xb + wb or y > yb + hb or y + h < yb:
            continue

        _, (width, height), _ = rect
        area = width * height
        boxes[(tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3]))] = area

        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        rectangular_contours.append(approx)


    sorted_boxes = sorted(boxes.items(), key=lambda x: x[1])

    warped = four_point_transform(clear_resized_image, sorted_boxes[0])

    cv.imshow('warped', warped)


    for rectangle in rectangular_contours:
        cv.drawContours(empty, [rectangle], -1, (255, 255, 255), 1)
        cv.drawContours(resized_image, [rectangle], -1, (255, 255, 255), cv.FILLED)




    # corners = cv.goodFeaturesToTrack(empty, 8, 0.01, 70)
    # corners = np.int0(corners)
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv.circle(resized_image, (x, y), 3, (0, 0, 255), -1)


    # rho = 1
    # theta = np.pi/180
    # threshold = 25
    # min_line_length = 200
    # max_line_gap = 25
    # lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    # print(lines)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(resized_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv.imshow('mask_contours', mask_contours)

    while True:
        #cv.imshow('mask', mask)
        cv.imshow('image', resized_image)
        key = cv.waitKey(0)
        if key == ord('a'):
            break

    print(f'image.shape: {image.shape}')
    # TODO: add image processing here
    return 'PO12345'