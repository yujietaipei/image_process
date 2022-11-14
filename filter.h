#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

//create ideal highpass filter with specified size
cv::Mat* ideal_highpass_filter(int width, int height, int radius);

//create ideal lowpass filter with specified size
cv::Mat* ideal_lowpass_filter(int height,int width,int radius);

//use filter to process the image in frequency domain
void image_process(cv::Mat& imag, const cv::Mat& filter);

#endif // FILTER_H
