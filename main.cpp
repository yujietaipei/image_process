#include <QApplication>
#include "fouriertransform.h"
#include "filter.h"

int main(int argc, char *argv[])
{
    //read the image and turn it into grayscale image
    cv::Mat img = cv::imread("C:/Qt/projects/practice/Lenna.png");
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //fourier transform on image
    cv::Mat* res = fft_2d(gray);

    //generate a ideal lowpass filter
    int radius = 80;
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
    QApplication a(argc, argv);
    //MainWindow w;
    //w.show();

    return a.exec();
}
