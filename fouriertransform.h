#ifndef FOURIERTRANSFORM_H
#define FOURIERTRANSFORM_H

#include <opencv2/opencv.hpp>
#include <vector>

//output the data in mat (only for uchar, double, and float)
void display(const cv::Mat& src);


//the fast fourier transform is based on radix-2 algorithm
//the reverse_num_list can calculate the order of index in first step
std::vector<unsigned int>* reverse_num_list(int length);

//the radix-2 algorithm is suitable for image with 2-based side length
//function find_length is to find the closest 2^n number
int find_length(int length);

//for complex number multiplication
cv::Vec2f comp_multiply(cv::Vec2f oper1,cv::Vec2f oper2);

//difference between fourier transform and inverse fourier transtorm is the sign of power of 'e'
//hence, the functions of both transforms are combined and separated by flag (1 for fft, 0 for ift)
//this function is for 1-dimensional transform, and the transform result is stored in source mat
//the shift is for handling the fft along vertical direction
void fft_ift_base(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set,bool  flag);

void fft_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set);

void ift_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set);

//fft_1d will allocate extra space to store transform result
cv::Mat* fft_1d(const cv::Mat& src);

//2-dimensional fourier transform (based on 1d fft along horizontal and vertical direction)
cv::Mat* fft_2d(cv::Mat& src);

//2-dimensional inverse fourier transform (based on 1d ift along horizontal and vertical direction)
void ift(cv::Mat& src);

#endif // FOURIERTRANSFORM_H
