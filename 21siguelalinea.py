# simple controller with onboard camera

import sys

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver

import numpy as np
import cv2
from datetime import datetime
import os
import time


# =========================================
# CONFIGURATION CONSTANTS
# =========================================

DEBOUNCE_TIME = 0.1

MAX_ANGLE = 0.10

MAX_SPEED = 80
SPEED_INCR = 5

ANGLE_INCR = 0.05


# =========================================
# PID VARIABLES
# =========================================

previous_error = 0
integral = 0


# =========================================
# GET IMAGE FROM CAMERA
# =========================================

def get_image(camera):

    raw_image = camera.getImage()

    image = np.frombuffer(
        raw_image,
        np.uint8
    ).reshape(
        (
            camera.getHeight(),
            camera.getWidth(),
            4
        )
    )

    return image


# =========================================
# HSV YELLOW MASK
# =========================================

def lane_mask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # =========================
    # YELLOW MASK
    # =========================

    lower_yellow = np.array([10, 80, 80])
    upper_yellow = np.array([45, 255, 255])

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # =========================
    # WHITE MASK
    # =========================

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])

    white_mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )

    # =========================
    # COMBINE MASKS
    # =========================

    combined_mask = cv2.bitwise_or(
        yellow_mask,
        white_mask
    )

    return combined_mask

# =========================================
# REGION OF INTEREST
# =========================================

def region_of_interest(image):

    height = image.shape[0]
    width = image.shape[1]

    polygons = np.array([
        [
            (0, height),
            (width, height),
            (width, int(height * 0.45)),
            (0, int(height * 0.45))
        ]
    ])

    mask = np.zeros_like(image)

    cv2.fillPoly(mask, polygons, 255)

    masked_image = cv2.bitwise_and(
        image,
        mask
    )

    return masked_image

# =========================================
# DRAW DETECTED LINES
# =========================================

def draw_lines(image, lines):

    line_image = np.zeros_like(image)

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            cv2.line(
                line_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

    combo_image = cv2.addWeighted(
        image,
        0.8,
        line_image,
        1,
        1
    )

    return combo_image


# =========================================
# CALCULATE ERROR
# =========================================

def calculate_error(
    lines,
    image_width,
    previous_error
):

    if lines is None:
        return 0

    setpoint = image_width / 2

    # IMPORTANTE
    smallest_error = None

    for line in lines:

        x1, y1, x2, y2 = line[0]

        # evitar división entre cero
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        line_length = np.sqrt(
            (x2 - x1)**2 +
            (y2 - y1)**2
        )

        if line_length < 15:
            continue

        print("Slope:", slope)

        # ignorar líneas horizontales
        if abs(slope) < 0.2:
            continue

        midpoint = (x1 + x2) / 2

        error = midpoint - setpoint

        # promedio de errores
        if smallest_error is None:
            smallest_error = error
        else:
            smallest_error = (
                smallest_error + error
            ) / 2

    print("Smallest error:", smallest_error)

    if smallest_error is None:
        return 0

    return smallest_error

# =========================================
# DISPLAY IMAGE
# =========================================

def display_image(display, image):

    image_rgb = np.dstack(
        (
            image,
            image,
            image
        )
    )

    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )

    display.imagePaste(
        image_ref,
        0,
        0,
        False
    )


# =========================================
# MAIN
# =========================================

def main():

    # PID gains
    kp = 0.008
    ki = 0
    kd = 0.002

    # Lower speed for stability
    speed = 80

    last_press = {}

    integral = 0
    previous_error = 0

    # Create robot
    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    # Camera
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Display
    display_img = Display("display_image")

    # Keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # =====================================
    # MAIN LOOP
    # =====================================

    while robot.step() != -1:

        # =================================
        # IMAGE ACQUISITION
        # =================================

        image = get_image(camera)

        frame = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )

        # =================================
        # IMAGE PROCESSING
        # =================================

        # Yellow detection
        lanes = lane_mask(frame)

        # Edge detection
        edges = cv2.Canny(
            lanes,
            50,
            150
        )

        # ROI
        cropped_edges = region_of_interest(edges)

        # Hough transform
        lines = cv2.HoughLinesP(
            cropped_edges,
            1,
            np.pi / 180,
            threshold=10,
            minLineLength=10,
            maxLineGap=50        )
        print("Lines:", lines)
        
        # Draw lines
        lane_image = draw_lines(
            frame,
            lines
        )

        # =================================
        # PID CONTROL
        # =================================

        error = calculate_error(
            lines,
            frame.shape[1],
            previous_error
        )

        if lines is None:
            speed = 15
        else:
            speed = 80

        print("Error:", error)

        # Integral
        integral += error

        # Derivative
        derivative = error - previous_error

        # PID
        steering = (
            kp * error
            + ki * integral
            + kd * derivative
        )

        # Steering limit
        steering = max(
            -0.10,
            min(0.10, steering)
        )

        print("Steering:", steering)

        # Apply steering
        driver.setSteeringAngle(
            steering
        )

        previous_error = error

        # =================================
        # DISPLAY IMAGE
        # =================================

        final_gray = cv2.cvtColor(
            lane_image,
            cv2.COLOR_BGR2GRAY
        )

        display_image(
            display_img,
            final_gray
        )

        # =================================
        # KEYBOARD CONTROL
        # =================================

        current_time = time.time()

        key = keyboard.getKey()

        if (
            key in last_press
            and (
                current_time
                - last_press[key]
                < DEBOUNCE_TIME
            )
        ):
            continue

        last_press[key] = current_time

        # Increase speed
        if key == keyboard.UP:

            if speed < MAX_SPEED:

                speed += SPEED_INCR

                print("Speed:", speed)

        # Decrease speed
        elif key == keyboard.DOWN:

            if speed >= SPEED_INCR:

                speed -= SPEED_INCR

                print("Speed:", speed)

        # Save image
        elif key == ord('A'):

            current_datetime = str(
                datetime.now().strftime(
                    "%Y-%m-%d %H-%M-%S"
                )
            )

            file_name = (
                current_datetime
                + ".png"
            )

            print("Image taken")

            camera.saveImage(
                os.getcwd()
                + "/"
                + file_name,
                1
            )

        # Apply speed
        driver.setCruisingSpeed(speed)


# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    main()