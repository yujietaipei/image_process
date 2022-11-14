#include "filter.h"

cv::Mat* ideal_highpass_filter(int width, int height, int radius){
    //the image is not shifted after fft, hence, the filter must be shifted
    cv::Mat* filter = new cv::Mat(height,width,CV_8UC1,cv::Scalar(1));
    int cx = width/2;
    int cy = height/2;

    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int dist_x = (i + cx)%width - cx;
            int dist_y = (j + cy)%height - cy;

            if(dist_x*dist_x + dist_y*dist_y < radius * radius){
                //if the cooresponding point in filter is in the cicle
                filter -> at<uchar>(i,j) = 0;
            }
        }
    }
    return filter;
}

cv::Mat* ideal_lowpass_filter(int height,int width,int radius){
    //the image is not shifted after fft, hence, the filter is shifted
    cv::Mat* filter = new cv::Mat(height,width,CV_32FC1,cv::Scalar(0));
    int cx = width/2;
    int cy = height/2;

    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int dist_x = (i + cx)%width - cx;
            int dist_y = (j + cy)%height - cy;
            if(dist_x*dist_x + dist_y*dist_y < radius * radius){
                //if the cooresponding point in filter is in the cicle
                filter -> at<float>(i,j) = 1;
            }
        }
    }
    return filter;
}

void image_process(cv::Mat& imag, const cv::Mat& filter){
    //check the size of imag and filter
    if(imag.rows != filter.rows){
        std::cerr << "the amount of rows is not matching";
        return;
    }
    if(imag.cols != filter.cols){
        std::cerr << "the amount of cols is not matching";
        return;
    }
    for(int i = 0;i < imag.rows;i++){
        for(int j = 0;j < imag.cols;j++){
            imag.at<cv::Vec2f>(i,j)[0] *= filter.at<float>(i,j);
            imag.at<cv::Vec2f>(i,j)[1] *= filter.at<float>(i,j);
        }
    }

}
