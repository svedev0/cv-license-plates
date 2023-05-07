import cv2
from pytesseract import pytesseract
import datetime

# Download Tesseract-OCR and place it in the root directory
# Download haarcascade_frontalface_default.xml and place it in the root directory

# Path to tesseract executable
pytesseract.tesseract_cmd = "Tesseract-OCR\\tesseract.exe"

log_file = "license_plates.txt"

# Create VideoCapture object for webcam
capture = cv2.VideoCapture(0)

# Loop until program is stopped
while True:
    # Read frame from webcam
    ret, frame = capture.read()

    # Convert frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur to reduce noise
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours in edges
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop over contours
    for contour in contours:
        # Get contour area
        area = cv2.contourArea(contour)

        # Discard small contours
        if area < 1000:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 50), 2)

        # Extract license plate region
        roi = grey[y : y + h, x : x + w]

        # Perform thresholding
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform dilation to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Perform OCR on license plate region
        plate_num = pytesseract.image_to_string(dilated, config="--psm 7")

        # Remove whitespace and non-alphanumeric characters
        plate_num = "".join(e for e in plate_num if e.isalnum())

        # If license plate number is detected, log it to file
        if len(plate_num) == 6:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, "a+") as f:
                f.write("{} {}\n".format(timestamp, plate_num))

    cv2.imshow("License Plate Recognition", frame)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
