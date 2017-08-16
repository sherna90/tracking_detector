//Author: Diego Vergara
#ifndef CPU_HAMILTONIAN_MC_H
#define CPU_HAMILTONIAN_MC_H
#include "CPU_logistic_regression.hpp"
#include "hmc.hpp"

class CPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, double _step_size = 0.01, int _num_step = 100, double _path_lenght = 0.0);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 0, bool erf = false, bool prob_label = false);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
protected:
 	CPU_LogisticRegression logistic_regression;
};

#endif // CPU_HAMILTONIAN_MC_H