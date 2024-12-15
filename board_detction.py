import numpy as np
import cv2
import matplotlib.pyplot as plt

OUTPUT_SIZE = 1448
 


def show_orginal_and_cropped(original, cropped):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original Image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Chessboard")
    axes[0].axis('off')
    
    # Cropped Image
    axes[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Cropped Chessboard")
    axes[1].axis('off')
    
    plt.show()


       

def get_contours(image, blur_radius):
    edge_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (blur_radius, blur_radius), 2)
    edge_image = cv2.Canny(edge_image, 20, 200)
    edge_image = cv2.dilate(edge_image, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)), iterations=1)
    contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    listContours = [cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True) for c in contours]
    contours = tuple(listContours)
    return contours, hierarchy

def get_angle(ab, ac):
    dot = ab.dot(ac)
    norm_prod = np.linalg.norm(ab) * np.linalg.norm(ac)
    cos_theta = np.clip(dot / norm_prod, -1, 1)
    theta = np.arccos(cos_theta)
    return np.degrees(theta) if not np.isnan(theta) else 0

def check_square(points):
    if len(points) != 4:
        return 4 * 90 ** 2, 0
    a, b, c, d = np.squeeze(points)
    angles = [
        get_angle(b - c, d - c),
        get_angle(c - d, a - d),
        get_angle(d - a, b - a),
        get_angle(a - b, c - b)
    ]
    return np.sum((np.array(angles) - 90) ** 2), np.mean(np.abs([a - b, b - c, c - d, d - a]))

def child_count(i, hierarchy, is_square):
    j = hierarchy[0, i, 2]
    total = 0
    while j >= 0:
        if is_square[j]:
            total += 1
        j = hierarchy[0, j, 0]
    return total

def get_candidate_boards(contours, hierarchy, max_error, min_side, min_child):
    square_info = list(map(lambda contour: check_square(contour), contours))
    is_square = list(map(lambda info: info[0] < max_error and info[1] > min_side, square_info))

    squares = []
    for i, (its_square, contour) in enumerate(zip(is_square, contours)):
        if its_square:  # big square
            total = child_count(i, hierarchy, is_square)  # little squares
            if total > min_child:  # Only consider squares with more than `min_child` children
                squares.append(contour)

                

    return squares


def preprocess(image, blur_radii=(3, 5, 7, 9, 11), max_error=400, min_side=10, min_child=10):
    all_boards = []
    for blur_radius in blur_radii:
        contours, hierarchy = get_contours(image, blur_radius)
        boards = get_candidate_boards(contours, hierarchy, max_error, min_side, min_child)
        all_boards.extend(boards)
    
    best_board = None
    best_side = 0
    for board in all_boards:
        error, side = check_square(board)
        if side > best_side:
            best_board = board
            best_side = side
    
    if best_board is not None and best_side >= 0:
        return best_board
    else:
        raise ValueError("No valid chessboard detected.")
    
    
def order_points(pts):
    pts = np.squeeze(pts)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect     
    

def crop_image(image, points):
    src_pts = order_points(points)
    dst_pts = np.array([[0, 0],
                        [OUTPUT_SIZE - 1, 0],
                        [OUTPUT_SIZE - 1, OUTPUT_SIZE - 1],
                        [0, OUTPUT_SIZE - 1]],
                        dtype="float32")
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed = cv2.warpPerspective(image,
                                      transform_matrix,
                                      (OUTPUT_SIZE, OUTPUT_SIZE))
    return transformed    


def divide_chessboard_into_squares(transformed):
    square_size = OUTPUT_SIZE // 8
    squares=[]
    
    for i in range(8):
        for j in range(8):

            x_start = j * square_size
            y_start = i * square_size
            x_end = x_start + square_size
            y_end = y_start + square_size
            
            # Extract the square from the chessboard image
            square = transformed[y_start:y_end, x_start:x_end]
            squares.append(square)
    
    return squares



def get_squares(image,show_cropped_image=False):
    board = preprocess(image, (5,7, 9, 11))
    cropped= crop_image(image, board)

    # Calculate and display centers on original image 
    squares = divide_chessboard_into_squares(cropped)

    if show_cropped_image:
        show_orginal_and_cropped(image, cropped)

    return squares

# # read and resize the image
# image=cv2.imread("v0.JPG")
# info=get_squares(image,True)

