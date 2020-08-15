#include <iostream>
#include <fstream>

#include "cu_utils.cuh"
#include "cu_cphys.cuh"

using namespace std;

void dumpState(const string& filename, double* state, long int timesteps, int bodies){
	ofstream stateFile;
	stateFile.open(filename.c_str());
	if(stateFile.is_open()){
		stateFile << bodies << "," << timesteps << "\n";
		stateFile.write((char*)state, 7*timesteps*bodies*sizeof(double));
	}
	stateFile.close();
}


double* load(const string& filename, int &size){
    double* mat = NULL;
    ifstream matFile;
    matFile.open(filename.c_str());

    if (matFile.is_open()) {
        matFile >> size;
        matFile.get();
        mat = new double [7*size];
        // Load all bytes in one step
        matFile.read((char*)mat, 7*size*sizeof(double));
    }
    matFile.close();

    return mat;
}

void print_state(double* state, int bodies){
	printf("X,Y,Z,Vx,Vy,Vz,M\n");
	for (int i=0; i < bodies; i++){
		for (int j = 0; j < 7; j++){
			printf("%f", state[i*7+j]);
			if (j != 6){
				printf(",");
			}
		}
		if (i != bodies-1){
			printf("\n");
		}
	}

}

void print_seven_vector(double* vec){
	cout << "<";
	for (int i = 0; i < 7; i++){
		cout << vec[i];
		if (i != 6){
			cout << ", ";
		}
	}
	cout << ">";
}

void arange(float low, float high, float step, long int &size){
	size = (long int)(high-low)/step;
	if (size*8 > 1000000000){
		cerr << "Error! You are trying to run a simulation which will take over a GB of memory to store.";
		cerr << " Please reduce integrateion time, or increase time step." << endl;
		cerr << "Total Requested Memory: " << size*8/1000000000.0 << " GB" << endl;
		exit(1);
	}
}

void errorCheck(int code, cudaError_t err)
{
    if(err != cudaSuccess) {
        printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
}

double* int_n_model(double* y0, double t, float h, int bodies, long int &timesteps){
    arange(0, t, h, timesteps);
    int panel_size = 7*bodies;

    if (7*bodies*timesteps*8 > 1000000000){
        cerr << "Error! You are trying to run a simulation which will take over a GB of memory to store.";
        cerr << " Please reduce integrateion time, or increase time step." << endl;
        cerr << "Total Requested Memory: " << 7*bodies*timesteps*8/1000000000.0 << " GB" << endl;
        exit(1);
    }

    double* pList;
    errorCheck(1, cudaMalloc((void **) &pList, bodies*7*sizeof(double)*timesteps));
    errorCheck(2, cudaMemcpy(pList, y0, bodies*7*sizeof(double), cudaMemcpyHostToDevice));

    int TILELENGTH = 10;
    dim3 dimGrid(ceil(bodies/(float)TILELENGTH), 1, 1);
    dim3 dimBlock(TILELENGTH, 1, 1);

    float progress;
    
    for (int i = 0; i < timesteps; i++){
        progress = ((float)i/timesteps)*100;
        cout << "Progress: " << int(progress) << "% Completed\r";
        cout.flush();
        time_step<<<dimGrid, dimBlock>>>(pList, h, bodies, i, panel_size, TILELENGTH);
        cudaDeviceSynchronize();
    }
    cout << endl;
    double* ys = new double [7*bodies*timesteps];
    errorCheck(3, cudaMemcpy(ys, pList, bodies*7*sizeof(double)*timesteps, cudaMemcpyDeviceToHost));

    return ys;
}
