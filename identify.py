import asyncio
import cv2
from pytesseract import pytesseract
import datetime

# Path to tesseract executable
pytesseract.tesseract_cmd = "Tesseract-OCR\\tesseract.exe"

LOG_FILE = "license_plates.txt"
CAPTURE = cv2.VideoCapture(0)


async def log_to_file(plate_num: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a+") as f:
        f.write("{} {}\n".format(timestamp, plate_num))


async def main():
    while CAPTURE.isOpened():
        ret, frame = CAPTURE.read()
        if not ret:
            continue

        # Greyscale and gaussian blur to reduce noise
        greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(greyscale, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(gaussian_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_color = (0, 255, 50)
            rect_width = 3
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, rect_width)

            license_plate_region = greyscale[y : y + h, x : x + w]
            _, thresh = cv2.threshold(
                license_plate_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Dilation to connect characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            plate_num_raw = pytesseract.image_to_string(dilated, config="--psm 7")
            plate_num = "".join(e for e in plate_num_raw if e.isalnum())

            if len(plate_num) == 6:
                # print(plate_num)
                task = asyncio.create_task(log_to_file(plate_num))
                await task

        cv2.imshow("License Plate Recognition", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    CAPTURE.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
