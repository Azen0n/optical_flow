import argparse

import numpy as np
import cv2

from horn_schunck import calc_image_derivatives


def lucas_kanade(first_image: np.ndarray, second_image: np.ndarray, kernel_radius: int = 1):
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

    dx, dy, dt = calc_image_derivatives(first_image, second_image)
    u = np.zeros(first_image.shape)
    v = np.zeros(first_image.shape)

    for i in range(kernel_radius, first_image.shape[0] - kernel_radius):
        for j in range(kernel_radius, first_image.shape[1] - kernel_radius):
            i_x = dx[i - kernel_radius:i + kernel_radius + 1, j - kernel_radius:j + kernel_radius + 1].flatten()
            i_y = dy[i - kernel_radius:i + kernel_radius + 1, j - kernel_radius:j + kernel_radius + 1].flatten()
            i_t = dt[i - kernel_radius:i + kernel_radius + 1, j - kernel_radius:j + kernel_radius + 1].flatten()
            a = np.column_stack((i_x, i_y))
            b = -i_t.reshape(-1, 1)
            u[i, j], v[i, j] = (1 / (a.T.dot(a))).dot(a.T).dot(b)
    return u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', help='set kernel radius', type=int)
    parser.add_argument('-v', '--video', help='set video path', type=str)
    args = parser.parse_args()

    if args.radius:
        kernel_radius = args.radius
    else:
        kernel_radius = 1

    if args.video:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture('video.mp4')

    while True:
        _, first_frame = capture.read()
        _, second_frame = capture.read()
        u, v = lucas_kanade(first_frame, second_frame, kernel_radius)
        cv2.imshow(f'Lucas-Kanade method, {kernel_radius = }', u + v)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
