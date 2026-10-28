import cv2
import numpy as np


def wait_and_destroy():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hu_log(hu7):
    # v_i = -sign(h_i) * log10(|h_i|)
    hu7 = np.array(hu7, dtype=np.float64).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.sign(hu7) * np.log10(np.abs(hu7))
    return v


def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


def solidity(cnt, area):
    hull = cv2.convexHull(cnt)
    ha = cv2.contourArea(hull)
    if ha <= 0:
        return 0.0
    return float(area) / float(ha)


def passes_shape_bands(hu1, hu2, area=None, circ=None, sol=None):
    """
    Improved acceptance logic:
    - Primary: expanded Hu1/Hu2 band from all observed animals
    - Fallback: accept if large area, high solidity, and low circularity (elongated animals)
    """
    eps = 1e-6
    # Expanded Hu ranges
    hu1_min, hu1_max = 0.17, 0.44
    hu2_min, hu2_max = 0.005, 0.14

    in_hu_band = (hu1_min <= hu1 <= hu1_max) and (hu2_min <= hu2 <= hu2_max)
    if in_hu_band:
        return True, "Animal"

    # Fallback rule
    if area is None:
        area = 0.0
    if circ is None:
        circ = 1.0
    if sol is None:
        sol = 0.0

    area_thresh = (
        1500  # Minimum allowed contour area in pixels — filters out tiny specks/noise
    )
    sol_thresh = 0.80  # Minimum solidity (area / convex hull area) — ensures the shape is mostly filled, not jagged
    circ_thresh = 0.22  # Maximum circularity (4π × area / perimeter²) — rejects very round shapes like stones or blobs

    fallback = (area >= area_thresh) and (sol >= sol_thresh) and (circ <= circ_thresh)
    if fallback:
        return True, "Animal"

    return False, "-"


#########################################################################################


def main():
    # Load image
    img = cv2.imread("../images/capture_savannah/img_8.jpg")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for green
    lower_green = np.array([40, 35, 10])  # Hue ~35–85 for green
    upper_green = np.array([85, 255, 255])

    # Create mask for green and invert
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    cv2.imwrite("../images/capture_savannah/mask.png", mask)
    cv2.imwrite("../images/capture_savannah/mask_inv.png", mask_inv)
    # Optional: clean up specks (helps stability)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, k, iterations=1)

    # Apply mask to remove grass
    result = cv2.bitwise_and(img, img, mask=clean)
    cv2.imwrite("../images/capture_savannah/without_grass.png", result)
    # Grayscale & binary
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite("../images/capture_savannah/greyscaled.png", thresh)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    min_area = 1500
    max_area = 500000
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # Sort and keep top N by area
    top_n = 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    # Additional priors
    circ_min, circ_max = 0.2, 0.60  # tune as needed
    sol_min = 0.60  # tune as needed
    draw_color_keep = (0, 0, 255)
    draw_color_drop = (0, 255, 255)

    contour_img = img.copy()
    rect_img = img.copy()
    for i, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Hu moments (linear + log)
        m = cv2.moments(cnt)
        hu = cv2.HuMoments(m).flatten()
        hu_log_vals = hu_log(hu)

        # Simple shape priors
        circ = circularity(area, perimeter)
        sol = solidity(cnt, area)

        # Core relation: (Hu1, Hu2) cluster bands
        in_band, band = passes_shape_bands(hu[0], hu[1], area=area, circ=circ, sol=sol)
        keep = in_band and (circ_min <= circ <= circ_max) and (sol >= sol_min)

        # Final keep rule
        keep = in_band and (circ_min <= circ <= circ_max) and (sol >= sol_min)

        print(
            f"\nContour {i}: A={area:.1f}, P={perimeter:.2f}, circ={circ:.3f}, sol={sol:.3f}"
        )
        print("  Hu (linear): " + ", ".join([f"{x:.6e}" for x in hu]))
        print("  Hu (log)   : " + ", ".join([f"{x:.3f}" for x in hu_log_vals]))
        print(f"  Band match : {band}")
        print(f"  -> keep    : {keep}")

        # Draw contour
        cv2.drawContours(
            contour_img, [cnt], -1, draw_color_keep if keep else draw_color_drop, 2
        )

        # Draw bounding rectangle if kept
        if keep:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), draw_color_keep, 2)
            cv2.putText(
                rect_img,
                f"{band}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                draw_color_keep,
                2,
                cv2.LINE_AA,
            )

    # Resize all images
    img = cv2.resize(img, (1500, 1000))
    contour_img = cv2.resize(contour_img, (1500, 1000))
    rect_img = cv2.resize(rect_img, (1500, 1000))
    result = cv2.resize(result, (1500, 1000))

    # Show results
    cv2.imshow("Original", img)
    wait_and_destroy()
    cv2.imshow("Without Grass (cleaned)", result)
    wait_and_destroy()
    cv2.imshow("Contours (keep=red, drop=yellow)", contour_img)
    cv2.imwrite("../images/capture_savannah/contours.png", contour_img)
    wait_and_destroy()
    cv2.imshow("Animals Highlighted", rect_img)
    cv2.imwrite("../images/capture_savannah/animals.png", rect_img)
    wait_and_destroy()


if __name__ == "__main__":
    main()
