import cv2
import time
import numpy as np
import keras
import RPi.GPIO as GPIO

# Khởi tạo GPIO
GPIO.setmode(GPIO.BCM)
relay_pin = 17
GPIO.setup(relay_pin, GPIO.OUT)

# Load the trained model
model = keras.models.load_model("e:/Desktop/fileloi.h5")

# Biến để theo dõi trạng thái trước đó của dự đoán
previous_prediction = None

# Function to capture and process image
def capture_and_predict():
    # Open a connection to the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            # Capture a frame
            ret, frame = cap.read()

            # Display the captured frame
            cv2.imshow("Camera Feed", frame)

            # Wait for 5 seconds
            time.sleep(5)

            # Save the captured image
            cv2.imwrite("captured_image.jpg", frame)

            # Preprocess the captured image for prediction
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_array = img / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions
            predictions = model.predict(img_array)

            # Determine the label based on predictions
            if predictions[0][0] > predictions[0][1]:
                label = 'Nắp công tơ chưa in'
            else:
                label = 'Nắp công tơ đã in'

            # Print and save the result
            print(label)
            with open("results.txt", "a", encoding="utf-8") as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                formatted_line = f"{timestamp} - {label}\n"
                f.write(formatted_line)

            # Check for a change in prediction
            if label != previous_prediction:
                # Change in prediction detected, control the relay
                if label == 'Nắp công tơ chưa in':
                    # Kích hoạt relay khi có sản phẩm lỗi
                    GPIO.output(relay_pin, GPIO.HIGH)
                else:
                    # Tắt relay khi sản phẩm không lỗi
                    GPIO.output(relay_pin, GPIO.LOW)

            # Update previous prediction
            previous_prediction = label

            # Add color differentiation based on predictions
            if predictions[0][0] > predictions[0][1]:
                print('Nắp công tơ chưa in')
                color = (0, 255, 0)  # Green color for "Nắp công tơ chưa in"
            else:
                print('Nắp công tơ đã in')
                color = (255, 255, 255)  # White color for "Nắp công tơ đã in"

            # Chuyển đổi sang không gian màu HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Đặt ngưỡng cho màu xanh
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)

            # Ngưỡng cho màu trắng
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([255, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            # Kết hợp mask xanh và mask trắng
            combined_mask = cv2.bitwise_or(mask_green, mask_white)

            # Áp dụng mask để loại bỏ nền
            result = cv2.bitwise_and(frame, frame, mask=combined_mask)

            # Hiển thị ảnh đã xử lý
            cv2.imshow("Processed Image", result)

            # Check if the 'q' key is pressed to exit the camera feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Release the camera and close windows when interrupted
        GPIO.cleanup()  # Cleanup GPIO pins
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_predict()
