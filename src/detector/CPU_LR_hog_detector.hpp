#ifndef CPU_LR_HOG_DETECTOR_H
#define CPU_LR_HOG_DETECTOR_H

#include "hog_detector.hpp"
#include <opencv2/ximgproc/segmentation.hpp>
#include "../likelihood/CPU_logistic_regression.hpp"
#include "../libs/piotr_fhog/fhog.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace cv::ximgproc::segmentation;

class CPU_LR_HOGDetector : public HOGDetector
{
public:
	void init(double group_threshold, double hit_threshold);
	vector<Rect> detect(Mat &frame);
	vector<double> detect(Mat &frame,vector<Rect> samples);
	void train(Mat &frame,Rect reference_roi);
	double train();
	VectorXd genHog(Mat &frame);
	VectorXd genRawPixels(Mat &frame);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	void generateFeatures(Mat &frame, int label);
	MatrixXd getFeatures();
protected:
	HOGDescriptor hog;
	CPU_LogisticRegression logistic_regression;
	int num_frame=0;
	double max_value=1.0;
	vector<Rect> sliding_window(Mat &frame,int stride);
	vector<Rect> region_proposal(Mat &frame);
	void data_clean();
};

#endif
