import argparse

import numpy as np
from scipy.ndimage import convolve
import cv2


def horn_schunck(first_image: np.ndarray, second_image: np.ndarray,
                 alpha: float = 1.0, number_of_iterations: int = 10) -> tuple:
    """Horn–Schunck method of estimating optical flow.

    first_image, second_image — two frames for optical flow estimation.
    alpha — regularization constant. Larger values lead to a smoother flow.
    number_of_iterations — number of iterations for laplacian approximation.
    returns flow vectors U, V."""
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

    u = np.zeros(first_image.shape)
    v = np.zeros(first_image.shape)

    dx, dy, dt = calc_image_derivatives(first_image, second_image)

    averaging_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                                 [1 / 6, 0, 1 / 6],
                                 [1 / 12, 1 / 6, 1 / 12]])

    for _ in range(number_of_iterations):
        u_averages = convolve(u, averaging_kernel)
        v_averages = convolve(v, averaging_kernel)

        derivative = (dx * u_averages + dy * v_averages + dt) / (alpha ** 2 + dx ** 2 + dy ** 2)
        u = u_averages - dx * derivative
        v = v_averages - dy * derivative

    return u, v


def calc_image_derivatives(first_image: np.ndarray, second_image: np.ndarray) -> tuple:
    """Calculate image derivatives along the x, y and time dimensions."""
    kernel_x = np.array([[-1, 1],
                         [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1],
                         [1, 1]]) * 0.25
    kernel_t = np.array([[1, 1],
                         [1, 1]]) * 0.25

    dx = convolve(first_image, kernel_x) + convolve(second_image, kernel_x)
    dy = convolve(first_image, kernel_y) + convolve(second_image, kernel_y)
    dt = convolve(first_image, kernel_t) + convolve(second_image, -kernel_t)

    return dx, dy, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', help='set regularization constant. Larger values lead to a smoother flow',
                        type=float)
    parser.add_argument('-i', '--iterations', help='set number of iterations for laplacian approximation',
                        type=int)
    parser.add_argument('-v', '--video', help='set video path', type=str)
    args = parser.parse_args()

    if args.alpha:
        alpha = args.alpha
    else:
        alpha = 20

    if args.video:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture('video.mp4')

    if args.iterations:
        number_of_iterations = args.iterations
    else:
        number_of_iterations = 2

    while True:
        _, first_frame = capture.read()
        _, second_frame = capture.read()
        u, v = horn_schunck(first_frame, second_frame, alpha, number_of_iterations)
        cv2.imshow(f'Horn-Schunck method, {alpha = }, it = {number_of_iterations}', u + v)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
