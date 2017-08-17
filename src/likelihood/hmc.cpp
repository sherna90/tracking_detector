//Author: Diego Vergara
#include "hmc.hpp"


Hamiltonian_MC::Hamiltonian_MC(){
	this->init_hmc = true;
}

void Hamiltonian_MC::acceptace_rate(){
	cout << "Acceptace Rate: "<< 100 * (float) this->accepted/this->sampled <<" %" <<endl;
}


VectorXd Hamiltonian_MC::initial_momentum(){
	return this->multivariate_gaussian.sample();
}

double Hamiltonian_MC::unif(double step_size){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.9, 1.1);
    return step_size * dis(gen);
}

VectorXd Hamiltonian_MC::random_generator(int dimension){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
  	normal_distribution<double> dnormal(0.0,1.0);
	VectorXd random_vector(dimension);

	for (int i = 0; i < dimension; ++i){
		random_vector(i) = dnormal(generator);
	}
	return random_vector;
}

double Hamiltonian_MC::random_uniform(){
	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

VectorXd Hamiltonian_MC::random_binomial(int n, VectorXd prob, int dim){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  	mt19937 generator;
  	generator.seed(seed1);
	VectorXd random_vector(dim);
	for (int i = 0; i < dim; ++i){
		binomial_distribution<int> dbinomial(n,prob(i));
		random_vector(i) = dbinomial(generator);
	}
	return random_vector;
}


double Hamiltonian_MC::avsigmaGauss(double mean, double var){
  double erflambda = sqrt(M_PI)/4;
  double out = 0.5+0.5*erf(erflambda*mean/sqrt(1+2*pow(erflambda,2)*var));
  return out;
}

VectorXd Hamiltonian_MC::cumGauss(VectorXd &w, MatrixXd &phi, MatrixXd &Smat){
  	int M = phi.rows();
  	VectorXd ptrain(M);
  	//VectorXd weights = w.tail(w.rows()-1);
  	//double bias = w(0);
  	//#pragma omp parallel for schedule(static)
	for (int i = 0; i < M; ++i){
	  double mean = w.dot(phi.row(i));
	  double var = (phi.row(i) * Smat * phi.row(i).transpose())(0);
	  ptrain(i) = avsigmaGauss(mean, var);
	}
    return ptrain;
}

MatrixXd Hamiltonian_MC::get_weights(){
	MatrixXd weights;

	if (this->init_hmc){	
		return this->weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return weights;
	}
}

void Hamiltonian_MC::set_weights(VectorXd &_weights){
	if (this->init_hmc){	
		this->mean_weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

void Hamiltonian_MC::set_weightsMatrix(MatrixXd &_weights){
	if (this->init_hmc){	
		this->weights = _weights;	
	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}


