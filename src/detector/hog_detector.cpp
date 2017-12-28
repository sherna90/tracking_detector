#include "hog_detector.hpp"
HOGDetector::HOGDetector(){
	
}


void HOGDetector::draw()
{
	for (size_t i = 0; i < this->detections.size(); i++)
    {
        Rect r = this->detections[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(this->frame, r.tl(), r.br(), cv::Scalar(255,0,0), 3);
    }
}

MatrixXd HOGDetector::getFeatures()
{
	return this->feature_values;
}

vector<double> HOGDetector::getWeights()
{
	return this->weights;
}

void HOGDetector::saveToCSV(string name, bool append){
	tools.writeToCSVfile(name+"_values.csv", this->feature_values, append);
	tools.writeToCSVfile(name+"_labels.csv", this->labels, append);
}

void HOGDetector::loadFeatures(MatrixXd features, VectorXd labels){
	this->feature_values = features;
	this->labels = labels;
}