#include "test_detector_cpu.hpp"

#ifndef PARAMS

const double GROUP_THRESHOLD = 0.0;
const double HIT_THRESHOLD = 0.5;
const double POSITIVE = 1.0;
const double NEGATIVE = 0.0;

#endif

TestDetector::TestDetector(){
	this->detector.init(GROUP_THRESHOLD, HIT_THRESHOLD);
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	this->generator.seed(seed1);
}

void TestDetector::generateFeatures(string train_path, string positive_list, string negative_list, string filename, double margin){
	string line;
	ifstream train_list((train_path+positive_list).c_str());
	if (!train_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	cout << "positive" << endl;
	bool append = false;
	while (getline(train_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		Rect centerROI(margin, margin, current_frame.cols - margin*2, current_frame.rows - margin*2);
		Mat croppedImage = current_frame(centerROI);
		this->detector.generateFeatures(croppedImage, POSITIVE);
		this->detector.saveToCSV(filename+"positive_MARS", append);
		this->detector.dataClean();
		append = true;
  	}
  	append = false;
	cout << "negative" << endl;
	ifstream test_list((train_path+negative_list).c_str());
	if (!test_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
  	while (getline(test_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		cout << img_path << endl;
		this->detector.generateFeatures(current_frame, NEGATIVE);
		this->detector.saveToCSV(filename+"negative_MARS", append);
		this->detector.dataClean();
		append = true;
  	}
  	cout << "finish" << endl;
}

void TestDetector::train(){
	chrono::time_point<std::chrono::system_clock> start, end;
	C_utils utils;
	string data_extension = "_values_"+to_string(0);
	string label_extension = "_labels_"+to_string(0);
	string positive_data_name = "MARS_DATA/train_positive";
	string negative_data_name = "MARS_DATA/train_negative";
	MatrixXd positive_data;
	VectorXd positive_labels;
	MatrixXd negative_data;
	VectorXd negative_labels;
	utils.read_Data(positive_data_name+data_extension,positive_data);
	utils.read_Labels(positive_data_name+label_extension,positive_labels);
	utils.read_Data(negative_data_name+data_extension,negative_data);
	utils.read_Labels(negative_data_name+label_extension,negative_labels);
	MatrixXd data_test(0, positive_data.cols());;
	VectorXd labels_test;
	data_test.resize(positive_data.rows()+negative_data.rows(), NoChange);
	labels_test.resize(positive_labels.rows()+negative_labels.rows());
	data_test << positive_data,negative_data;
	labels_test << positive_labels,negative_labels;
	bool data_processing=true;
	for(int epochs=0;epochs<100;epochs++){
		for(int num_batches=1;num_batches<184;num_batches++){
			start = std::chrono::system_clock::now();
			data_extension = "_values_"+to_string(num_batches);
			label_extension = "_labels_"+to_string(num_batches);
			utils.read_Data(positive_data_name+data_extension,positive_data);
		 	utils.read_Labels(positive_data_name+label_extension,positive_labels);
		 	utils.read_Data(negative_data_name+data_extension,negative_data);
		 	utils.read_Labels(negative_data_name+label_extension,negative_labels);
			MatrixXd data(0, positive_data.cols());;
		 	VectorXd labels;
		 	data.resize(positive_data.rows()+negative_data.rows(), NoChange);
			labels.resize(positive_labels.rows()+negative_labels.rows());
			data << positive_data,negative_data;
			labels << positive_labels,negative_labels;
			cout << "-----------------------------------------------------------------------------------------" << endl;
			utils.dataPermutation(data, labels);
			this->detector.loadFeatures(data, labels);
			double loss=this->detector.train();
			VectorXd predicted_labels=this->detector.predictTest(data_test,data_processing);
			double accuracy=utils.calculateAccuracyPercent(labels_test,predicted_labels);
			end = std::chrono::system_clock::now();
      		int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds> (end-start).count();
			cout << " Epoch : " << epochs  << " | Mini Batch : "<<  to_string(num_batches) << " | Train Loss : " << loss << " | Test Error : " << accuracy << " | Ellapsed Time: " << elapsed_seconds << "[s]" << endl;
			data_processing=false;
			//utils.report(labels_test, predicted_labels, false);
			//utils.calculateAccuracyPercent(labels_test, predicted_labels);
			//utils.confusion_matrix(labels_test, predicted_labels, true);
		}
	}

}

void TestDetector::test_detector(string test_path, string positive_list, string negative_list){
	cout << "Positive" << endl;
	double positive_percent = this->detect(test_path, positive_list);
	cout << "Accuracy: "<< positive_percent << endl;
	cout << "Negative" << endl;	
	double negative_percent = this->detect(test_path, negative_list);
	cout << "Accuracy: "<< negative_percent << endl;
	
};


void TestDetector::loadModel(){
	C_utils utils;
	VectorXd mean;
	VectorXd std;
	VectorXd max;
	VectorXd min;
	VectorXd weights;
	VectorXd bias;
	utils.read_Labels("Model_means.csv",mean);
	utils.read_Labels("Model_weights.csv",weights);
	utils.read_Labels("Model_stds.csv",std);
	utils.read_Labels("Model_bias.csv",bias);
	utils.read_Labels("Model_maxs.csv",max);
	utils.read_Labels("Model_mins.csv",min);
	this->detector.loadModel(weights,mean,std, max, min, bias(0));
};
 
double TestDetector::detect(string train_path, string list){
	string line;
	ifstream detect_list((train_path+list).c_str());
	if (!detect_list) CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
	vector<Rect> detections;
	namedWindow("Detector");
	while (getline(detect_list, line)) {
		string img_path = train_path+line;
		Mat current_frame = imread(img_path);
		Mat grayImg;
		detections.clear();
		//int newHeight = 200;
    	//int newWidth = current_frame.cols*newHeight/current_frame.rows;
		//resize(current_frame, current_frame, Size(newWidth, newHeight));
	    detections = this->detector.detect(current_frame);
		vector<double> weights = this->detector.getWeights(); 
		for (int i = 0; i < detections.size(); ++i){
				Rect r=detections.at(i);
				rectangle( current_frame,r, Scalar(0,0,255), 2, LINE_8  );
				rectangle( current_frame,Point(r.x,r.y-10),Point(r.x+r.width,r.y+20),Scalar(0,0,255), -1,8,0 );
				putText(current_frame,to_string(weights.at(i)),Point(r.x+5,r.y+12),FONT_HERSHEY_SIMPLEX,0.5,Scalar(255,255,255),1);
		}
		imshow("Detector", current_frame);
		waitKey(0);

  	}
  	return 0;
};



int main(int argc, char* argv[]){
	
	string test_path = string("/media/data/data/MOT2017/train/MOT17-11-LRHOG/");
	string train_path = string("MARS_DATA/MARS/");
	string positive_list = string("pos.lst");
	string negative_list = string("neg.lst");


	TestDetector tracker = TestDetector();
	//tracker.generateFeatures(train_path, positive_list, negative_list, "train_", 0);
	//tracker.loadModel();
	tracker.train();
	//tracker.test_detector(train_path, positive_list, negative_list);
	//tracker.detect(test_path,positive_list);
	//tracker.test();
}