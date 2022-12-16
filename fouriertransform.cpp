#include "fouriertransform.h"

//struct for thread function params
typedef struct tData{
    cv::MatIterator_<cv::Vec2f> iterator;             //point to the first element of subimage
    int shift;                                                                    //the distance between each element in 1-d transformation
    int step;                                                                    //the distance between each layer in 1-d transformation
    int layers;                                                                 //the amount of rows/cols in a thread
    std::vector<unsigned int>* idx_set;                 //the reverse bit order number list for fourier transformation
}tData, *TDATA;

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
std::vector<unsigned int>* reverse_num_list(int length){

    //check the size of image is power of 2
    unsigned int  count = 0;
    unsigned int  len = length;
    while(len){
        if((len & 1) == 1){
            count++;
        }
        len  >>=  1;
    }
    if(count !=1){
        throw  std::invalid_argument("the length has to be the power of 2");
    }

    unsigned int digits = __builtin_clz(length) + 1;

    std::vector<unsigned int>* idx = new std::vector<unsigned int>(length,0);
    //array idx is used to store the order of index in the first step of fourier transform

    for(unsigned int i = 0;i < length; i++){
        unsigned int num = i;
        num = (num & 0xffff0000) >> 16 | (num & 0x0000ffff) << 16;
        num = (num & 0xff00ff00) >> 8 | (num & 0x00ff00ff) << 8;
        num = (num & 0xf0f0f0f0)  >> 4 | (num & 0x0f0f0f0f) << 4;
        num = (num & 0xcccccccc) >> 2 | (num & 0x33333333) << 2;
        num = (num & 0xaaaaaaaa) >> 1 | (num & 0x55555555) << 1;
        num >>= digits;
        idx -> at(i) = num;
        }

    return idx;
}

//the radix-2 algorithm is suitable for image with 2-based side length
//function find_length is to find the closest 2^n number
int find_length(int length){
    unsigned int tmp = length;
    int count = 0; // count the amount of 1
    while(tmp){
        if(tmp & 1){
            count++;
        }
    }
    if(count == 1){
        return length;
    }
    else{
        return 1 << (32 - __builtin_clz(length));
    }
}

//for complex number multiplication
cv::Vec2f comp_multiply(cv::Vec2f oper1,cv::Vec2f oper2){
    return cv::Vec<float, 2>( oper1[0]*oper2[0]- oper1[1]*oper2[1], oper1[0]*oper2[1] + oper1[1]*oper2[0]);
}

//difference between fourier transform and inverse fourier transtorm is the sign of power of 'e'
//hence, the functions of both transform are combined and separated by flag (1 for fft, 0 for ift)
//this function is for 1-dimensional transform, and the transform result is stored in source mat
//the shift is for handling the fft along vertical direction
void fft_ift_base(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<unsigned int>* idx_set,bool  flag){
    unsigned int length = idx_set -> size();
    const double PI = atan(1.0)*4;
    int group_size = 2;         //the dft length in each steps and it will grow on 2 base

    //Mat "from" and "to" store the original value and transformation result, and their functions are switched to reduce extra space
    cv::Mat from(1,length,CV_32FC2,cv::Scalar(0,0));
    cv::Mat to(1,length,CV_32FC2,cv::Scalar(0,0));
    cv::Mat from_ptr = from;
    cv::Mat to_ptr = to;

    //load the element from src based on the order in idx_set
    for(int i = 0; i< length;i++){
        cv::MatConstIterator_<cv::Vec2f> curr = it + (*idx_set)[i]*shift;
        from.at<cv::Vec2f>(i) = *curr;
    }

    while(group_size >> 1  != length){
        int offset = group_size >> 1;
        for(int i = 0;i < length;i++){
            int idx1 = i;
            int idx2 = (i + offset) % group_size + i/group_size * group_size;
            double tmp = (flag ? -1: 1)*2*i*PI/group_size;
            cv::Mat factor(1,1,CV_32FC2,cv::Scalar((float)cos(tmp),(float)sin(tmp)));
            to_ptr.at<cv::Vec2f>(i) = from_ptr.at<cv::Vec2f>(std::min(idx1,idx2))
                    + comp_multiply(from_ptr.at<cv::Vec2f>(std::max(idx1,idx2)),factor.at<cv::Vec2f>(0));
        }
        group_size <<= 1;

        //swap the pointer of to and from to use same space repeatedly
        swap(to_ptr,from_ptr);
    }

    //copy result to source mat
    for(int i = 0;i < length;i++){
        *it = from_ptr.at<cv::Vec2f>(i);
        it += shift;
    }
}



void fft_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<unsigned int>* idx_set){
        fft_ift_base(it,shift,idx_set,1);
   }

void ift_1d_inplace(cv::MatIterator_<cv::Vec2f> it, int shift, const std::vector<unsigned int>* idx_set){
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
    //convert the original image into float type
    cv::Mat* real = new cv::Mat_<float>(src);
    //store the imaginary part of original image
    cv::Mat* imag = new cv::Mat(src.size(),CV_32F, cv::Scalar(0));
    cv::Mat channels [2] = {*real, *imag};
    cv::Mat* transformResult = new cv::Mat(src.rows, src.cols, CV_32FC2, cv::Scalar(0));
    merge(channels, 2 , *transformResult);

    int MAX_LAYER = 250;                                                   //maximum processing row/column for a thread
    int restRows = transformResult -> rows;               //rest rows for processing
    int restCols = transformResult -> cols;                   // rest cols for processing

    cv::MatIterator_<cv::Vec2f> it = transformResult -> begin<cv::Vec2f>();

    //calculate the amount of thread (a thread transform MAX_LAYER rows at most)
    size_t ROW_MAX_THREAD = ceil(static_cast<double>(transformResult -> rows) / MAX_LAYER);
    size_t COL_MAX_THREAD = ceil(static_cast<double>(transformResult -> cols) / MAX_LAYER);

    //allocate memory to store thread id, handle and thread arguments
    DWORD* ROW_THREAD_ID = new DWORD[ROW_MAX_THREAD];
    HANDLE* ROW_THREAD_HANDLE = new HANDLE[ROW_MAX_THREAD];
    TDATA* ROW_THREAD_DATA = new TDATA[ROW_MAX_THREAD];

    DWORD* COL_THREAD_ID = new DWORD[COL_MAX_THREAD];
    HANDLE* COL_THREAD_HANDLE = new HANDLE[COL_MAX_THREAD];
    TDATA* COL_THREAD_DATA = new TDATA[COL_MAX_THREAD];

    std::vector<unsigned int>* row_idx_set = reverse_num_list(transformResult -> cols); //reverse bits list for horizontal transform
    std::vector<unsigned int>* col_idx_set = reverse_num_list(transformResult -> rows); //reverse bits list for vertical transform

    //transform along horizontal direction
    for(int i = 0;i < ROW_MAX_THREAD;i++){
        //allocate memory for thread arguments
        ROW_THREAD_DATA[i] = new struct tData;

        if(ROW_THREAD_DATA[i] == nullptr){
            std::cerr << "thread data allocation failure";
            exit(EXIT_FAILURE);
        }

        ROW_THREAD_DATA[i] -> iterator = it;                                                                                                               // first element in subimage in a thread
        ROW_THREAD_DATA[i] -> shift = 1;                                                                                                                      // the distance of each element in image
        ROW_THREAD_DATA[i] -> step = transformResult -> step / transformResult -> elemSize();           // the distance of each row
        ROW_THREAD_DATA[i] -> layers = (restRows < MAX_LAYER ? restRows : MAX_LAYER);                   //how many layers are distributed to a thread
        ROW_THREAD_DATA[i] -> idx_set = row_idx_set;
        it += ROW_THREAD_DATA[i] -> step * ROW_THREAD_DATA[i] -> layers;                                                 // calculate the iterator to the first element of next thread
        restRows -= ROW_THREAD_DATA[i] -> layers;


        //create thread
        ROW_THREAD_HANDLE[i] = CreateThread(NULL, 0, threadFunction, ROW_THREAD_DATA[i], 0, &ROW_THREAD_ID[i]);

        if(ROW_THREAD_HANDLE[i] == nullptr){
            std::cerr << "thread creation failure";
            exit(EXIT_FAILURE);
        }
    }

    if(restRows){
        std::cerr << "error occurs when distrubute subimage to thread <row>";
    }

    if(it != transformResult -> end<cv::Vec2f>()){
        std::cerr << "error in iterator manupulation <row>";
    }

        //calling thread won't resume until  all new thread terminates
        WaitForMultipleObjects(ROW_MAX_THREAD, ROW_THREAD_HANDLE, TRUE, INFINITE);

        //close all handles and allocated memory for thread
        for(int i = 0; i<ROW_MAX_THREAD;i++){
            CloseHandle(ROW_THREAD_HANDLE[i]);
            if(ROW_THREAD_DATA[i] != nullptr){
                delete ROW_THREAD_DATA[i];
            }
        }
        delete[] ROW_THREAD_HANDLE;
        delete[] ROW_THREAD_ID;
        delete[] ROW_THREAD_DATA;

        //transform along vertical direction, the implementation is the similar to transform along horizontal direction
        it = transformResult -> begin<cv::Vec2f>();
        for(int i = 0;i < COL_MAX_THREAD;i++){
            COL_THREAD_DATA[i] = new struct tData;

            if(COL_THREAD_DATA[i] == nullptr){
                std::cerr << "thread data allocation failure";
                exit(EXIT_FAILURE);
            }

            COL_THREAD_DATA[i] -> iterator = it;
            COL_THREAD_DATA[i] -> shift = transformResult -> step / transformResult -> elemSize();
            COL_THREAD_DATA[i] -> step = 1;
            COL_THREAD_DATA[i] -> layers = (restCols < MAX_LAYER ? restCols : MAX_LAYER);
            COL_THREAD_DATA[i] -> idx_set = col_idx_set;
            it += COL_THREAD_DATA[i] -> step * COL_THREAD_DATA[i] -> layers;
            restCols -= COL_THREAD_DATA[i] -> layers;

            COL_THREAD_HANDLE[i] = CreateThread(NULL, 0, threadFunction, COL_THREAD_DATA[i], 0, &COL_THREAD_ID[i]);

            if(COL_THREAD_HANDLE[i] == nullptr){
                std::cerr << "thread creation failure";
                exit(EXIT_FAILURE);
            }
        }

        if(restCols){
            std::cerr << "error occurs when distrubute subimage to thread <col> ";
        }


        WaitForMultipleObjects(COL_MAX_THREAD, COL_THREAD_HANDLE, TRUE, INFINITE);

        //close all handles and allocated memory for thread
        for(int i = 0; i<COL_MAX_THREAD;i++){
            CloseHandle(COL_THREAD_HANDLE[i]);
            if(COL_THREAD_DATA[i] != nullptr){
                delete COL_THREAD_DATA[i];
            }
        }
        delete[] COL_THREAD_HANDLE;
        delete[] COL_THREAD_ID;
        delete[] COL_THREAD_DATA;

    return transformResult;
}

//2-dimensional inverse fourier transform (based on 1d ift along horizontal and vertical direction)
void ift(cv::Mat& src){
    if(src.type() != CV_32FC2){
        std::cerr << "the data type cannot be transformed";
        return;
    }

    int MAX_LAYER = 250;                            //maximum processing row/column for a thread
    int restRows = src. rows;                       //rest rows for processing
    int restCols = src.cols;                            // rest cols for processing

    cv::MatIterator_<cv::Vec2f> it = src.begin<cv::Vec2f>();

    //calculate the amount of thread (a thread transform MAX_LAYER rows at most)
    size_t ROW_MAX_THREAD = ceil(static_cast<double>(src. rows) / MAX_LAYER);
    size_t COL_MAX_THREAD = ceil(static_cast<double>(src.cols) / MAX_LAYER);

    //allocate memory to store thread id, handle and thread arguments
    DWORD* ROW_THREAD_ID = new DWORD[ROW_MAX_THREAD];
    HANDLE* ROW_THREAD_HANDLE = new HANDLE[ROW_MAX_THREAD];
    TDATA* ROW_THREAD_DATA = new TDATA[ROW_MAX_THREAD];

    DWORD* COL_THREAD_ID = new DWORD[COL_MAX_THREAD];
    HANDLE* COL_THREAD_HANDLE = new HANDLE[COL_MAX_THREAD];
    TDATA* COL_THREAD_DATA = new TDATA[COL_MAX_THREAD];

    std::vector<unsigned int>* row_idx_set = reverse_num_list(src.cols); //reverse bits list for horizontal transform
    std::vector<unsigned int>* col_idx_set = reverse_num_list(src.rows); //reverse bits list for vertical transform
    //transform along horizontal direction
    for(int i = 0;i < ROW_MAX_THREAD;i++){
        //allocate memory for thread arguments
        ROW_THREAD_DATA[i] = new struct tData;

        if(ROW_THREAD_DATA[i] == nullptr){
            std::cerr << "thread data allocation failure";
            exit(EXIT_FAILURE);
        }

        ROW_THREAD_DATA[i] -> iterator = it;
        ROW_THREAD_DATA[i] -> shift = 1;
        ROW_THREAD_DATA[i] -> step = src.step /src.elemSize();
        ROW_THREAD_DATA[i] -> layers = (restRows < MAX_LAYER ? restRows : MAX_LAYER);
        ROW_THREAD_DATA[i] -> idx_set = row_idx_set;
        it += ROW_THREAD_DATA[i] -> step * ROW_THREAD_DATA[i] -> layers;
        restRows -= ROW_THREAD_DATA[i] -> layers;


        //create thread
        ROW_THREAD_HANDLE[i] = CreateThread(NULL, 0, inverseThreadFunction, ROW_THREAD_DATA[i], 0, &ROW_THREAD_ID[i]);

        if(ROW_THREAD_HANDLE[i] == nullptr){
            std::cerr << "thread creation failure";
            exit(EXIT_FAILURE);
        }
    }

    if(restRows){
        std::cerr << "error occurs when distrubute subimage to thread <row>";
    }

        //calling thread won't resume until  all new thread terminates
        WaitForMultipleObjects(ROW_MAX_THREAD, ROW_THREAD_HANDLE, TRUE, INFINITE);

        //close all handles and allocated memory for thread
        for(int i = 0; i<ROW_MAX_THREAD;i++){
            CloseHandle(ROW_THREAD_HANDLE[i]);
            if(ROW_THREAD_DATA[i] != nullptr){
                delete ROW_THREAD_DATA[i];
            }
        }
        delete[] ROW_THREAD_HANDLE;
        delete[] ROW_THREAD_ID;
        delete[] ROW_THREAD_DATA;

        it = src.begin<cv::Vec2f>();
        for(int i = 0;i < COL_MAX_THREAD;i++){
            COL_THREAD_DATA[i] = new struct tData;

            if(COL_THREAD_DATA[i] == nullptr){
                std::cerr << "thread data allocation failure";
                exit(EXIT_FAILURE);
            }

            COL_THREAD_DATA[i] -> iterator = it;
            COL_THREAD_DATA[i] -> shift = src. step /src.elemSize();
            COL_THREAD_DATA[i] -> step = 1;
            COL_THREAD_DATA[i] -> layers = (restCols < MAX_LAYER ? restCols : MAX_LAYER);
            COL_THREAD_DATA[i] -> idx_set = col_idx_set;
            it += COL_THREAD_DATA[i] -> step * COL_THREAD_DATA[i] -> layers;
            restCols -= COL_THREAD_DATA[i] -> layers;

            COL_THREAD_HANDLE[i] = CreateThread(NULL, 0, inverseThreadFunction, COL_THREAD_DATA[i], 0, &COL_THREAD_ID[i]);

            if(COL_THREAD_HANDLE[i] == nullptr){
                std::cerr << "thread creation failure";
                exit(EXIT_FAILURE);
            }


        }

        if(restCols){
            std::cerr << "error occurs when distrubute subimage to thread <col> ";
        }


        WaitForMultipleObjects(COL_MAX_THREAD, COL_THREAD_HANDLE, TRUE, INFINITE);

        //close all handles and allocated memory for thread
        for(int i = 0; i<COL_MAX_THREAD;i++){
            CloseHandle(COL_THREAD_HANDLE[i]);
            if(COL_THREAD_DATA[i] != nullptr){
                delete COL_THREAD_DATA[i];
            }
        }
        delete[] COL_THREAD_HANDLE;
        delete[] COL_THREAD_ID;
        delete[] COL_THREAD_DATA;

    //divide all the element by rows* cols
    int factor = src.rows* src.cols;
    for(int i = 0;i < src.rows;i++){
        for(int j = 0; j < src.cols;j++){
            src.at<cv::Vec2f>(i,j)[0] /= factor;
            src.at<cv::Vec2f>(i,j)[1] /= factor;
        }
    }

}


DWORD WINAPI threadFunction(LPVOID lpParam){
    TDATA param = (TDATA)lpParam;
    cv::MatIterator_<cv::Vec2f> it = param -> iterator;
    for(int i = 0;i < param -> layers;i++){
        fft_1d_inplace(it, param -> shift, param -> idx_set);
        it += param -> step;
    }
    return 0;
}

DWORD WINAPI inverseThreadFunction(LPVOID lpParam){
    TDATA param = (TDATA)lpParam;
    cv::MatIterator_<cv::Vec2f> it = param -> iterator;
    for(int i = 0;i < param -> layers;i++){
        ift_1d_inplace(it, param -> shift, param -> idx_set);
        it += param -> step;
    }
    return 0;
}
