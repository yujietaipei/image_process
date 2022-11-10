#include "mainwindow.h"

#include <QApplication>
#include <opencv2/opencv.hpp>
#include <vector>

//output the data in mat (only for uchar, double, and float)
void display(const cv::Mat& src){
    //for uchar
    if(src.type() == CV_8UC1){
        for(int i = 0;i < src.rows;i++){
            for(int j = 0;j < src.cols;j++){
                std::cout << (int)src.at<uchar>(i,j)<< " ";
            }
            std::cout << std::endl;
        }
    }
    //for double
    else if(src.type() == CV_64FC2){
        for(int i = 0;i < src.rows;i++){
            for(int j = 0;j < src.cols;j++){
                std::cout << src.at<cv::Vec2d>(i,j)[0]<< " + (" << src.at<cv::Vec2d>(i,j)[1] << ")i\t";
            }
            std::cout << std::endl;
        }
    }
    //for float
    else if(src.type() == CV_32FC2){
        for(int i = 0;i < src.rows;i++){
            for(int j = 0;j < src.cols;j++){
                std::cout << src.at<cv::Vec2f>(i,j)[0]<< " + (" << src.at<cv::Vec2f>(i,j)[1] << ")i\t";
            }
            std::cout << std::endl;
        }
    }
    else{
        std::cerr<<"the function is not applicable to data type";
    }
    std::cout << std::endl;
}

//the fast fourier transform is based on radix-2 algorithm
//the reverse_num_list can calculate the order of index in first step
std::vector<int>* reverse_num_list(int length){

    if(length == 1){
        std::vector<int>* idx = new std::vector<int>;
        idx -> push_back(0);
        return idx;
    }

    int digits = 0;
    //calculate the amount of bits to express numbers[0, length)
    int tmp = length;
    while(tmp != 1){
        digits++;
        tmp /= 2;
    }

    std::vector<bool> bits(digits,0);
    std::vector<int>* idx = new std::vector<int>(length,0);
    //array idx is used to store the order of index in the first step of fourier transform

    for(int i = 0;i < length; i++){
        int num = 0;

        for(int j = 0;j < digits;j++){
            num = num*2 + bits[j];
        }
        idx -> at(i) = num;

        //plus one to the reversed bit expression
        int k = 0;
        bool carry = 1;
        while(carry != 0 && k < bits.size()){
            if(bits[k] == 0){
                bits[k] = 1;
                carry = 0;
            }
            else{
                bits[k] = 0;
                carry = 1;
                k++;

            }

        }

    }
    return idx;

}

//the radix-2 algorithm is suitable for image with 2-based side length
//function find_length is to find the closest 2^n number
int find_length(int length){
    int res = 1;
    while(res < length){
        res *= 2;
    }
    return res;
}

//for complex number multiplication
cv::Vec2f comp_multiply(cv::Vec2f oper1,cv::Vec2f oper2){
    cv::Vec2f res;
    res[0] = oper1[0]*oper2[0]- oper1[1]*oper2[1];
    res[1] = oper1[0]*oper2[1] + oper1[1]*oper2[0];
    return res;
}

//difference between fourier transform and inverse fourier transtorm is the sign of power of 'e'
//hence, the functions of both transform are combined and separated by flag (1 for fft, 0 for ift)
//this function is for 1-dimensional transform, and the transform result is stored in source mat
//the shift is for handling the fft along vertical direction
void fft_ift_base(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set,bool  flag){
    int length = idx_set -> size();
    const double PI = atan(1.0)*4;
    int group_size = 2;         //the dft length in each steps and it will grow on 2 base

    //Mat "from" and "to" store the original value and transformation result, and their functions are switched reduce extra space
    cv::Mat from(1,length,CV_32FC2,cv::Scalar(0,0));
    cv::Mat to(1,length,CV_32FC2,cv::Scalar(0,0));
    cv::Mat from_ptr = from;
    cv::Mat to_ptr = to;

    //load the element from src based on the order in idx_set
    for(int i = 0; i< length;i++){
        cv::MatConstIterator_<cv::Vec2f> curr = it + (*idx_set)[i]*shift;
        from.at<cv::Vec2f>(i) = *curr;
    }

    while(group_size/2 != length){
        int offset = group_size/2;
        for(int i = 0;i < length;i++){
            int idx1 = i;
            int idx2 = (i + offset) % group_size + i/group_size * group_size;
            double tmp = (flag ? -1: 1)*2*i*PI/group_size;
            cv::Mat factor(1,1,CV_32FC2,cv::Scalar((float)cos(tmp),(float)sin(tmp)));
            to_ptr.at<cv::Vec2f>(i) = from_ptr.at<cv::Vec2f>(std::min(idx1,idx2))
                    + comp_multiply(from_ptr.at<cv::Vec2f>(std::max(idx1,idx2)),factor.at<cv::Vec2f>(0));
        }
        group_size *= 2;

        //swap the pointer of to and from to use same space repeatedly
        swap(to_ptr,from_ptr);
    }

    //copy result to source mat
    for(int i = 0;i < length;i++){
        *it = from_ptr.at<cv::Vec2f>(i);
        it += shift;
    }
}

void fft_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set){
        fft_ift_base(it,shift,idx_set,1);
   }

void ift_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<int>* idx_set){
        fft_ift_base(it,shift,idx_set,0);
}

//fft_1d will allocate extra space to store transform result
cv::Mat* fft_1d(const cv::Mat& src){
    if(src.empty()){
        std::cerr << "Error occurs when passing mat to fft_1d" << std::endl;
    }

    cv::Mat* result = new cv::Mat(src.size(),CV_32FC2,cv::Scalar(0,0));
    for(int i = 0;i < src.cols;i++){
        result -> at<cv::Vec2f>(i)[0] = src.at<uchar>(i);
    }
    cv::MatIterator_<cv::Vec2f> it = result -> begin<cv::Vec2f>();
    fft_1d_inplace(it,1,reverse_num_list(src.cols));

    return result;
}

//2-dimensional fourier transform (based on 1d fft along horizontal and vertical direction)
cv::Mat* fft_2d(cv::Mat& src){
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat* res = new cv::Mat; //store transform result

    //fourier transform in horizontal direction
    for(int i = 0;i < rows;i++){
        cv::Mat tmp = src.row(i);
        if(tmp.empty()){
            std::cout << "Error occurs in get the header of a certain row";
        }
        cv::Mat* subres = fft_1d(src.row(i));
        res -> push_back(*subres);
    }

    //fourier transform in vertical direction
    std::vector<int>* idx_set2 = reverse_num_list(rows);
    cv::MatIterator_<cv::Vec2f> it = res -> begin<cv::Vec2f>();
    for(int j = 0;j < cols;j++){
        fft_1d_inplace(it,res -> cols,idx_set2);
        it++;
    }
    return res;
}

//2-dimensional inverse fourier transform (based on 1d ift along horizontal and vertical direction)
void ift(cv::Mat& src){
    if(src.type() != CV_32FC2){
        std::cerr << "the data type cannot be transformed";
        return;
    }

    //inverse fourier transform in horizontal direction
    std::vector<int>* idx_set1 = reverse_num_list(src.cols);
    cv::MatIterator_<cv::Vec2f> row = src.begin<cv::Vec2f>();
    for(int i = 0;i < src.rows;i++){
        ift_1d_inplace(row,1,idx_set1);
        row += src.cols;
    }

    //inverse fourier transform in vertical direction
    std::vector<int>* idx_set2 = reverse_num_list(src.rows);
    cv::MatIterator_<cv::Vec2f>col = src.begin<cv::Vec2f>();
    for(int j = 0 ; j < src.cols ; j++){
        ift_1d_inplace(col,src.cols,idx_set2);
        col++;
     }

    //divide all the element by rows* cols
    int factor = src.rows* src.cols;
    for(int i = 0;i < src.rows;i++){
        for(int j = 0; j < src.cols;j++){
            src.at<cv::Vec2f>(i,j)[0] /= factor;
            src.at<cv::Vec2f>(i,j)[1] /= factor;
        }
    }
}

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

int main(int argc, char *argv[])
{
    //read the image and turn it into grayscale image
    cv::Mat img = cv::imread("C:/Qt/projects/practice/Lenna.png");
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //fourier transform on image
    cv::Mat* res = fft_2d(gray);

    //generate a ideal lowpass filter
    int radius = 100;
    cv::Mat* filter = ideal_lowpass_filter(gray.rows,gray.cols,radius);

    //process the image
    image_process(*res,*filter);

    //inverse fourier transform
    ift(*res);

    //convert the data type from float to uchar
    cv::Mat newimage;
    res -> convertTo(newimage,CV_8UC2);

    //split the newimage into real and imag part
    cv::Mat channels [2];
    split(newimage,channels);

    //display result
    cv::imshow("Sub Image after transform",channels[0]);


    return 0;
}
