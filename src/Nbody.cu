#include<iostream>
#include<string>
#include<fstream>
#include<math.h>

#include "cu_utils.cuh"

using namespace std;


int main(int argc, char *argv[]){
	int bodies;

	if(argc != 5){
		cerr << "Please call program as $ ./a.out inputFile outputFile ttotal dt." << endl;
		exit(1);
	}

	string filename=argv[1];

	double* init_conditions;

	init_conditions = load(filename, bodies);

	cout << "Number of Bodies is: " << bodies << endl;

	
	long int timesteps;
	double *ys;
	ys = int_n_model(init_conditions, strtod(argv[3], NULL), atof(argv[4]), bodies, timesteps);

	cout << "Writing to File, this may take a moment" << endl;
	string outputFile = argv[2];
	dumpState(outputFile, ys, timesteps, bodies);

	delete[] init_conditions;
	delete[] ys;

	return 0;
}
