#include "hazeremoval.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;
int main() {
    int num_iterations = 100;
    double total_cpu_time = 0.0;

    for (int i = 0; i < num_iterations; i++) {
        clock_t start, end;
        double cpu_time_used;
        start = clock();

        const char* img_path = "F:/photo/suong1.jpg";
        Mat in_img = imread(img_path);

        Mat out_img(in_img.rows, in_img.cols, CV_8UC3);
        unsigned char* indata = in_img.data;
        unsigned char* outdata = out_img.data;

        CHazeRemoval hr;
        cout << hr.InitProc(in_img.cols, in_img.rows, in_img.channels()) << endl;
        cout << hr.Process(indata, outdata, in_img.cols, in_img.rows, in_img.channels()) << endl;

        cv::imwrite("F:/photo/ketqua2/khu" + to_string(i + 1) + ".jpg", out_img);

        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        total_cpu_time += cpu_time_used;

        cout << "Iteration " << (i + 1) << " CPU time: " << cpu_time_used << " s" << endl;
    }

    double average_cpu_time = total_cpu_time / num_iterations;
    cout << "Average CPU time: " << average_cpu_time << " s" << endl;

    return 0;
}