# function [result] = fft2_image(in_img)
#
# %in_img = imread('D:\python_code_folder\word_test\samsung\samsung_data_2\train\c5kf2l97yg.jpg');
#
# in_img_gray = im2gray(in_img);
#
# in_img_gray_fft2 = fft2(in_img_gray);
#
# in_img_gray_fft2_abs = fftshift(in_img_gray_fft2);
#
# in_img_gray_fft2_abs = abs(in_img_gray_fft2_abs);
#
# %result = mat2gray(in_img_gray_fft2_abs);
# result = mat2gray(log(in_img_gray_fft2_abs));