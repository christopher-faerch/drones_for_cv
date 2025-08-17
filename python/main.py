import cv2
from matplotlib.pyplot import draw
import numpy as np
from src.plotter import Plotter
from src.vision import Vision

def show_hsv_picker(img_bgr):
    # Convert to HSV once
    if img_bgr is None:
        return
    height, width, _ = img_bgr.shape
    if height > 1080 or width == 1920:
        img_bgr = cv2.resize(img_bgr, (width // 4, height // 4))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click
            l, a, b = lab[y, x]
            r = 5  # 5 -> 11x11 window
            y0 = max(0, y - r)
            y1 = min(lab.shape[0], y + r + 1)  # +1 because slicing end is exclusive
            x0 = max(0, x - r)
            x1 = min(lab.shape[1], x + r + 1)

            surrounding_pixels = lab[y0:y1, x0:x1]  # shape ~ (<=11, <=11, 3) near borders
            print(f"Surrounding pixels (L, A, B): {surrounding_pixels}")
            print(f"Pixel at ({x}, {y}) - L: {l}, A: {a}, B: {b}")

    cv2.namedWindow("Click to see LAB")
    cv2.setMouseCallback("Click to see LAB", mouse_callback)

    while True:
        cv2.imshow("Click to see LAB", lab)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()

def main():

    vision = Vision()
    vision.load_image("capture_6/img_0.jpg")
    # vision.draw_penis()
    # vision.save_color_channels()
    # vision.save_upper_half()
    # vision.save_hsv_image()
    # vision.brightest_pixel()
    # vision.load_flower_image()
    # vision.find_yellow_hsv()
    # vision.find_yellow_lab()
    # plotter = Plotter()
    # plotter.plot_histogram(vision.get_img())
    # plotter.plot_histogram_g(vision.get_img())
    # vision.get_each_pixel_green()
    # vision.find_contours(vision.get_binary_mask())
    vision.mahalanobis_model("python/images/capture_6/img_0")
    vision.apply_mahalanobis()
    # img = cv2.imread("python/images/capture_5/img_8.jpg")
    # annot = cv2.imread("python/images/capture_6/img_0_annotated.jpg")   # you painted the unwanted pixels

    # keep_mask, d2_img, filtered = vision.mahalanobis_filter(
    #     img, annot,
    #     colorspace='Lab', use_channels=(1,2),   # a,b only (robust to lighting)
    #     method='percentile', q=0.98             # cover 98% of your annotated cluster
    # )

    # cv2.imwrite("keep_mask.png", keep_mask)
    # cv2.imwrite("filtered.png", filtered)
    # show_hsv_picker(vision.get_img())

if __name__ == "__main__":
    main()
    