#include "filter.h"
//the image is not centered after transform, and hence, the filter is not centered, either

cv::Mat* build_filter(int height, int width, std::vector<int>&  params, float(*fun)(int , int ,std::vector<int>&) ){
    cv::Mat* filter = new cv::Mat(height,width,CV_32FC1,cv::Scalar(0));
    int cx = width >> 1;
    int cy = height >> 1;
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            int dist_x = (i + cx)%width - cx;
            int dist_y = (j + cy)%height - cy;
            filter -> at<float>(i,j) = fun(dist_x,dist_y,params);
        }
    }
    return filter;
}

float ideal_highpass(int disX, int disY, std::vector<int>& params){
    int cutOff = params[0];
    return (disX * disX + disY * disY < cutOff * cutOff ? 0 : 1);
}

cv::Mat* ideal_highpass_filter(int height, int width, int cutOff){
    std::vector <int> params{cutOff};
    return build_filter(height, width, params, ideal_highpass);
}

float gaussian_highpass(int disX, int disY,std::vector<int>& params){
    //disX and distY is the distance from center to specific coordination
    int cutOff = params[0];
    return 1 - exp(-static_cast<float>(disX*disX + disY*disY)/(2 * cutOff * cutOff));
}

cv::Mat* gaussian_highpass_filter(int height, int width, int cutOff){
    std::vector<int> params {cutOff};
    return build_filter(height, width, params, gaussian_highpass);
}

float butterworth_highpass(int disX, int disY, std::vector<int>& params){
    int cutOff = params[0];
    int pow_n = params[1];
    return 1 / (1 + pow(static_cast<float>(cutOff * cutOff)/(disX * disX + disY * disY), pow_n));
}

cv::Mat* butterworth_highpass_filter(int height, int width, int cutOff, int pow_n){
    std::vector<int> params {cutOff, pow_n};
    return build_filter(height, width, params, butterworth_highpass);
}

float ideal_lowpass(int disX, int disY, std::vector<int>& params){
    int cutOff = params[0];
    return (disX * disX + disY * disY < cutOff * cutOff ? 1 : 0);
}

cv::Mat* ideal_lowpass_filter(int height,int width,int cutOff){
    std::vector<int> params{cutOff};
    return build_filter(height, width,params,ideal_lowpass);
}

float gaussin_lowpass(int disX, int disY, std::vector<int>& params){
    int cutOff = params[0];
    return exp(-static_cast<float>(disX * disX + disY * disY)/(2 * cutOff*cutOff));
}

cv::Mat* gaussian_lowpass_filter(int height, int width, int cutOff){
    std::vector<int> params{cutOff};
    return build_filter(height,width,params,gaussin_lowpass);
}

float butterworth_lowpass(int disX, int disY, std::vector<int>& params){
    int cutOff = params[0];
    int pow_n = params[1];
    return 1 / (1 + pow(static_cast<float>(disX * disX + disY * disY)/(cutOff*cutOff), pow_n));
}

cv::Mat* butterworth_lowpass_filter(int height, int width, int cutOff, int pow_n){
    std::vector<int> params {cutOff, pow_n};
    return build_filter(height, width, params, butterworth_lowpass);
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
