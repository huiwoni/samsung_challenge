# matlab code

# listing = dir('D:\python_code_folder\word_test\samsung\samsung_data_2\test');
#
# num = length(listing);
#
# for i = 3:num
# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
# s = strcat('D:\python_code_folder\word_test\samsung\samsung_data_2\test\',listing(i).name);
# in_img = imread(s);
#
# [result] = fft2_image(in_img);
# % result = result * 10000;
# % result = mat2gray(result);
# result_uint8 = uint8(result * 255);
#
# out_img = in_img;
# out_img(:,:, 1) = result_uint8;
# out_img(:,:, 2) = result_uint8;
# out_img(:,:, 3) = result_uint8;
#
# result_s = strcat('D:\python_code_folder\word_test\samsung\samsung_data_2\result_test_log\',listing(i).name);
# imwrite(out_img, result_s);
# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
# end