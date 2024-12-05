import cv2
import numpy as np
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import threading


def process_superpixel(i, labels, original, lock, result_text):
    # Create a mask for the current superpixel
    superpixel_mask = (labels == i).astype(np.uint8) * 255

    # Apply the mask to extract the region
    superpixel_region = cv2.bitwise_and(
        original, original, mask=superpixel_mask)

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(superpixel_region, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(gray, config="--psm 6").strip()

    # Append text with thread safety
    if text:
        with lock:
            result_text.append(f"Superpixel {i}:\n{text}\n\n")

    print(f'Finished {i}')


def superpixel_to_text_parallel(image_path, region_size=30, ruler=10.0):
    # Load image
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert to LAB color space for SLIC
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Create SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(
        lab_image, algorithm=cv2.ximgproc.SLIC, region_size=region_size, ruler=ruler)
    slic.iterate(10)

    # Get superpixel labels
    labels = slic.getLabels()
    num_superpixels = slic.getNumberOfSuperpixels()

    # Lock for thread-safe text updates
    lock = threading.Lock()
    result_text = []

    # Process superpixels in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_superpixel, i, labels,
                                   original, lock, result_text) for i in range(num_superpixels)]
        for i, future in enumerate(futures):
            print(f'Processing {i+1}/{num_superpixels}')

    return ''.join(result_text), image


# Test the function
image_path = "imagenes/best_summer_ever.jpeg"  # Replace with your image path
text, visualized_image = superpixel_to_text_parallel(image_path)

# Save the visualization
cv2.imwrite("superpixel_visualization.jpg", visualized_image)

# Print the extracted text
print("Extracted Text:")
print(text)
