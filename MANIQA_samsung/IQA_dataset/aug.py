import os
import cv2
import numpy as np
from random import randint, uniform

# 디렉터리 설정
directory = './train'
output_directory = './train_aug'

# 디렉터리의 파일 목록 가져오기
listing = os.listdir(directory)

# 파일 반복 처리
for filename in listing:
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        s = os.path.join(directory, filename)
        imOriginal = cv2.imread(s)

        # Apply random transformations
        for i in range(1, 6):
            imAugmented = imOriginal.copy()

            if i == 1:
                angle = randint(-10, 10)
                rows, cols, _ = imAugmented.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                imAugmented = cv2.warpAffine(imAugmented, M, (cols, rows))

            elif i == 2:
                x_translation = randint(-50, 50)
                y_translation = randint(-50, 50)
                M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
                imAugmented = cv2.warpAffine(imAugmented, M, (imAugmented.shape[1], imAugmented.shape[0]))
            elif i == 3:
                scale_factor = uniform(1.2, 1.5)
                rows, cols, _ = imAugmented.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale_factor)
                imAugmented = cv2.warpAffine(imAugmented, M, (cols, rows))

            elif i == 4:
                imAugmented = cv2.flip(imAugmented, -1)  # 상하 및 좌우 반전
            elif i == 5:
                shear_angle = uniform(0, 0.1)
                M = np.float32([[1, shear_angle, 0], [0, 1, 0]])
                imAugmented = cv2.warpAffine(imAugmented, M, ((imAugmented.shape[1]+int(abs(imAugmented.shape[1]*shear_angle))), imAugmented.shape[0]))
            result_s = os.path.join(output_directory, f'{i}_{filename}')
            cv2.imwrite(result_s, imAugmented)