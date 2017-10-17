#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc/segmentation.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include "../utils/c_utils.hpp"
#include "../DPP/nms.hpp"
#include "../DPP/dpp.hpp"
#include "../likelihood/logistic_regression.hpp"
#include "../likelihood/CPU_logistic_regression.hpp"
#include "../libs/piotr_fhog/fhog.hpp"


struct Args {
	bool make_gray = true;
    bool resize_src = true;
    int width, height;
    int hog_width, hog_height;
    double scale;
    int nlevels;
    double gr_threshold;
    double hit_threshold;
    bool hit_threshold_auto;
    int win_width;
    int test_stride_width, test_stride_height;
    int train_stride_width, train_stride_height;
    int block_width;
    int block_stride_width, block_stride_height;
    int cell_width;
    int nbins;
    bool gamma_corr;
    double overlap_threshold;
    double p_accept;
    double lambda, epsilon, tolerance;
    int n_iterations;
    int padding;
} ;

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::ximgproc::segmentation;

class CPU_LR_HOGDetector 
{
public:
	void init(double group_threshold, double hit_threshold);
	vector<Rect> detect(Mat &frame);
	void train();
	VectorXd getFeatures(Mat &frame);
    vector<double> getWeights();
	void generateFeatures(Mat &frame, double label);
	void dataClean();
	void draw();
	void saveToCSV(string name, bool append = true);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	void loadFeatures(MatrixXd features, VectorXd labels);
	
protected:
	Args args;
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
	int num_frame=0;
	double max_value=1.0;
	MatrixXd feature_values;
	int group_threshold;
	double hit_threshold;
	int n_descriptors, n_data;
	vector<Rect> detections;
	VectorXd labels;
	vector<double> weights;
	Mat frame;
	C_utils tools;
	mt19937 generator;

};

#endif
