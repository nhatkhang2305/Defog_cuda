#ifndef HAZE_REMOVAL_H
#define HAZE_REMOVAL_H

#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <vector>

typedef struct _pixel {
	int i;
	int j;
	uchar val;
	_pixel(int _i, int _j, uchar _val) :i(_i), j(_j), val((uchar)_val) {}
} Pixel;

class CHazeRemoval
{
public:
	CHazeRemoval();
	~CHazeRemoval();

public:
	bool InitProc(int width, int height, int nChannels);
	bool Process(const unsigned char* indata, unsigned char* outdata, int width, int height, int nChannels);

private:
	int rows;
	int cols;
	int channels;

};

void get_dark_channel(const cv::cuda::GpuMat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius);

void get_air_light(const cv::cuda::GpuMat *p_src, std::vector<Pixel> &tmp_vec, cv::Vec3d *p_Alight, int rows, int cols, int channels);

void get_transmission(const cv::cuda::GpuMat *p_src, cv::cuda::GpuMat *p_tran, cv::Vec3d *p_Alight, int rows, int cols, int channels, int radius, double omega);

void recover(const cv::cuda::GpuMat *p_src, const cv::Mat* p_tran, cv::Mat* p_dst, cv::Vec3d* p_Alight, int rows, int cols, int channels, double t0);
void assign_data(unsigned char *outdata, const cv::cuda::GpuMat *p_dst, int rows, int cols, int channels);

#endif // !HAZE_REMOVAL_H