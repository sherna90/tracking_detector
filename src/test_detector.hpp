#ifndef TEST_DETECTOR_H
#define TEST_DETECTOR_H

//#include "detector/cpu_hog_detector.hpp"
#include "detector/cuda_hog_detector.hpp"
#include "utils/c_utils.hpp"
#include "utils/image_generator.hpp"

#include <time.h>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

class TestDetector{
public:
	TestDetector();
    void generateFeatures(string train_path, string positive_list, string negative_list, string filename, double margin);
    void train();
    void test_detector(string train_path, string positive_list, string negative_list);
    void test();
    double detect(string train_path, string list);
    void loadModel();
private:
	vector<Mat> images;
	vector<string> gt_vec;
	//CPU_HOGDetector detector;
    CUDA_HOGDetector detector;
    mt19937 generator;
};

#endif