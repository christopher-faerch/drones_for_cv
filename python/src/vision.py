import cv2
from matplotlib.pyplot import draw
import numpy as np
import os 

path = "python/images/"

class Vision:

    def __init__(self, original_image = "img/original_field.jpg"):
        self.original_image = original_image
        self.img = None
        self.mahalanobis_distance = None

    def get_img(self):
        return self.img

    def load_image(self, image_name: str) -> None:
        self.img = cv2.imread(os.path.join(path, image_name))
        if self.img is None:
            print("Error: Image not found.")

    def show_image(self, img: np.array) -> None:
        if img is None:
            print("Error: No image to display.")
            return

        # Detect shape
        if img.ndim == 2:  # Grayscale
            height, width = img.shape
        elif img.ndim == 3:  # Color
            height, width, _ = img.shape
        else:
            print("Error: Unsupported image shape.")
            return

        # Resize if larger than Full HD
        if height > 1080 and width > 1920:
            img = cv2.resize(img, (width // 4, height // 4))

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def make_empty_image(self) -> None:
        self.img = np.zeros((100, 200, 3), np.uint8)
        cv2.line(self.img, (20, 30), (40, 120),(0, 0, 255), 3)
        cv2.imwrite(os.path.join(path, "test.png"), self.img)


    def get_binary_mask(self) -> np.ndarray:
        if self.img is None:
            print("Error: No image provided.")
            return None

        # BGR -> Lab
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)

        # Split and take ONLY the A channel (index 1)
        _, A, _ = cv2.split(lab)

        # Keep values INSIDE [118, 255] on the A channel (inclusive)
        a_min, a_max = 118, 255
        mask = cv2.inRange(A, a_min, a_max)   # white where A in [118,255], black otherwise

        # Optional smoothing of the binary mask
        gauss_mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Apply mask to original image (for visualization)
        filtered = cv2.bitwise_and(self.img, self.img, mask=gauss_mask)
        cv2.imwrite(os.path.join(path, "[1.6.8]_Filter_A_118-255.png"), filtered)

        # Return the binary mask (smoothed)
        return gauss_mask

    def find_contours(self, binary_mask: np.ndarray) -> np.ndarray:
        if binary_mask is None:
            print("Error: No binary mask provided.")
            return []

        # Ensure single-channel uint8
        if binary_mask.ndim == 3:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        if binary_mask.dtype != np.uint8:
            binary_mask = (binary_mask > 0).astype(np.uint8) * 255

        # Detect edges
        edges = cv2.Canny(binary_mask, 100, 200)
        if not np.any(edges):
            print("No edges found.")
            return edges

        cnts_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

        # Nothing found?
        if not contours:
            print("No contours found.")
            self.show_image(self.img if self.img is not None else edges)
            return edges

        # OPTIONAL: ignore tiny specks
        min_area = 250  # tweak or set to 0 to disable
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not contours:
            print("No contours above min_area.")
            self.show_image(self.img if self.img is not None else edges)
            return edges

        # Keep only the largest N contours
        top_k = -1  # change to 3, 5, ... if you want more
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_k]

        # Draw on a copy of the original image
        vis = self.img.copy() if self.img is not None else cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 5)

        self.show_image(vis)
        return contours

    def print_hsv_values(self, pixel_start_location, pixel_end_location) -> None:
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        h_values, s_values, v_values = [], [], []

        for y in range(pixel_start_location[1], pixel_end_location[1] + 1):
            for x in range(pixel_start_location[0], pixel_end_location[0] + 1):
                h, s, v = hsv_img[y, x]
                h_values.append(h)
                s_values.append(s)
                v_values.append(v)
                print(f"Pixel at ({x}, {y}) - H: {h}, S: {s}, V: {v}")

        # Compute statistics
        def stats(values):
            return min(values), max(values), sum(values) / len(values)

        h_min, h_max, h_avg = stats(h_values)
        s_min, s_max, s_avg = stats(s_values)
        v_min, v_max, v_avg = stats(v_values)

        print("\n=== HSV Statistics in Range ===")
        print(f"Hue       -> Min: {h_min}, Max: {h_max}, Avg: {h_avg:.2f}")
        print(f"Saturation-> Min: {s_min}, Max: {s_max}, Avg: {s_avg:.2f}")
        print(f"Value     -> Min: {v_min}, Max: {v_max}, Avg: {v_avg:.2f}")
        print("\n=== HSV Statistics ===")

    def mahalanobis_model(self, file_name, q=0.98, morph=3):
        # Read images
        img_path   = file_name + ".jpg"
        annot_path = file_name + "_annotated.jpg"
        img        = cv2.imread(img_path)
        img_annot  = cv2.imread(annot_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {os.path.abspath(img_path)}")
        if img_annot is None:
            raise FileNotFoundError(f"Could not read annotated image: {os.path.abspath(annot_path)}")

        # 1) Find your painted annotations (bright red marks in BGR)
        lower_limit = (0, 0, 245)
        upper_limit = (10, 10, 255)  # <= 255 (fixed)
        mask = cv2.inRange(img_annot, lower_limit, upper_limit)

        # 2) Build the cluster from annotated pixels (unwanted class of pixels)
        pixels = img.reshape(-1, 3).astype(np.float32)
        mask_pixels = mask.reshape(-1)
        annot_pix_values = pixels[mask_pixels == 255]
        if annot_pix_values.shape[0] < 10:
            raise ValueError("Not enough annotated pixels to estimate mean/cov.")

        avg = annot_pix_values.mean(axis=0)
        cov = np.cov(annot_pix_values, rowvar=False)

        # Regularize covariance to avoid singular inverse
        eps = 1e-6 * np.trace(cov) / cov.shape[0]
        cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
        inv_cov = np.linalg.inv(cov)

        # 3) Mahalanobis distance^2 for all pixels (your formulation)
        diff = pixels - avg
        mahalanobis_dist2 = np.sum(diff * (diff @ inv_cov), axis=1)
        mahalanobis_distance_image = mahalanobis_dist2.reshape(img.shape[0], img.shape[1]).astype(np.float32)
        self.mahalanobis_distance = mahalanobis_distance_image

        # 4) Choose cutoff from your annotated cluster (q-percentile)
        diff_annot = annot_pix_values - avg
        d2_annot = np.sum(diff_annot * (diff_annot @ inv_cov), axis=1)
        T2 = float(np.quantile(d2_annot, q))   # e.g., q=0.98 keeps 98% of the cluster

        # --- minimal addition: store model for reuse ---
        self.maha_avg = avg.astype(np.float32)
        self.maha_inv_cov = inv_cov.astype(np.float32)
        self.maha_T2 = T2
        # also persist to disk so another function/run can use it
        model_path = os.path.join(path, "mahalanobis_model.npz")
        np.savez_compressed(model_path, avg=self.maha_avg, inv_cov=self.maha_inv_cov, T2=self.maha_T2)

        # 5) Pixels similar to annotations (d2 ≤ T2) are "unwanted"
        unwanted = (self.mahalanobis_distance <= T2)
        keep_mask = (~unwanted).astype(np.uint8) * 255

        # Optional cleanup
        if morph and morph > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
            keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_OPEN, k)
            keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_CLOSE, k)

        # 6) Apply mask
        filtered = cv2.bitwise_and(img, img, mask=keep_mask)

        # Save
        cv2.imwrite(os.path.join(os.path.dirname(file_name), "mahalanobis_filtered.png"), filtered)
        cv2.imwrite(os.path.join(os.path.dirname(file_name), "mahalanobis_mask.png"), keep_mask)

        return keep_mask, filtered


    def apply_mahalanobis(self, morph=3):
        if self.img is None:
            raise FileNotFoundError(f"Could not read image")

        # Load model from memory if available; else from disk
        if not hasattr(self, "maha_inv_cov"):
            model_path = os.path.join(path, "mahalanobis_model.npz")
            data = np.load(model_path)
            self.maha_avg = data["avg"].astype(np.float32)
            self.maha_inv_cov = data["inv_cov"].astype(np.float32)
            self.maha_T2 = float(data["T2"])

        pixels = self.img.reshape(-1, 3).astype(np.float32)
        diff = pixels - self.maha_avg
        d2 = np.sum(diff * (diff @ self.maha_inv_cov), axis=1).astype(np.float32)
        d2_img = d2.reshape(self.img.shape[:2])

        keep_mask = (d2_img > self.maha_T2).astype(np.uint8) * 255

        if morph and morph > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
            keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_OPEN, k)
            keep_mask = cv2.morphologyEx(keep_mask, cv2.MORPH_CLOSE, k)

        filtered = cv2.bitwise_and(self.img, self.img, mask=keep_mask)
        cv2.imwrite(os.path.join(path, "Filtered_image_after_mahalanobis.png"), filtered)
        # Find contours on the filtered image
        contours = self.find_contours(keep_mask)

        # Draw the first contour
        if contours:
            contours_single = contours[0] if contours else None
            cv2.drawContours(self.img, [contours_single], 0, (255, 0, 0), 2)
            self.show_image(self.img)

        hu_moments = []
        for contour in contours:
            hu_moments.append(self.hu_moments(contour))

        return keep_mask, contours

    def hu_moments(self, contour: list):
        moments = cv2.moments(contour)
        hu_features = cv2.HuMoments(moments).flatten()
        print("Hu Moments:", hu_features)
        return hu_features