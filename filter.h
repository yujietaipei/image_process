#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

//function to build the filter (width, height) is the size of the filter
//params is the parameters for filter function (Gaussian, ideal etc.)
//fun is funcion pointer of filter function
cv::Mat* build_filter(int width, int height, std::vector<int>& params, float(*fun)(int, int, std::vector<int>&) );

//XXX_highpass and XXX_lowpass are functions of filter
//XXX_highpass_filter and XXX_lowpass_filter are filter generation functions (for users)
float ideal_highpass(int disX, int disY, std::vector<int>& params);

cv::Mat* ideal_highpass_filter(int width, int height, int cutOff);

float gaussian_highpass(int disX, int disY, std::vector<int> & params);

cv::Mat* gaussian_highpass_filter(int width, int height, int cutOff);

float butterworth_highpass(int disX, int disY, std::vector<int>& params);

cv::Mat* butterworth_highpass_filter(int height, int width, int cutOff, int pow_n);

float ideal_lowpass(int disX, int disY, std::vector<int>& params);

cv::Mat* ideal_lowpass_filter(int height,int width,int cutOff);

float gaussin_lowpass(int disX, int disY, std::vector<int>& params);

cv::Mat* gaussian_lowpass_filter(int height, int width, int cutOff);

float butterworth_lowpass(int disX, int disY, std::vector<int>& params);

cv::Mat* butterworth_lowpass_filter(int height, int width, int cutOff, int pow_n);

//use filter to process the image in frequency domain
void image_process(cv::Mat& imag, const cv::Mat& filter);

#endif // FILTER_H
