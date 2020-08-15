#include <iostream>
#include <string>
#include <math.h>
#include <cstdlib>

#include "cu_cphys.cuh"
#include "cu_nbmath.cuh"

using namespace std;


__device__ void model(double* dydt, double* y0, int ID, double* pList, int bodies){
	double G = 6.6*pow((float)10, -11);
	double* r = new double [3];
	double rmag = 0;
	double rtemp = 0;

	for (int i = 0; i < 3; i++){
		dydt[i] = y0[i+3];
		dydt[i+3] = 0;
		r[i] = 0;
	}
	dydt[6] = 0;

	for (int body = 0; body < bodies; body++){
		rmag = 0;
		if (ID != body){
			for (int i = 0; i < 3; i++){
				// pList[body*7+i] for i in range(0, 3) access the position vec
				//      for each body
				rtemp = pList[body*7+i]-y0[i];
				rmag += rtemp*rtemp;
				r[i] = rtemp;
			}
			rmag = sqrt(rmag);
			if (rmag > 0.5){
				for (int i = 3; i < 6; i++){
					// pList[body*7+6] access the mass for each body
					dydt[i] += ((G*pList[body*7+6])/((rmag*rmag)))*(r[i-3]/rmag);
				}
				
			}
		}
	}
	delete[] r;
}



__global__ void time_step(double* ys, float h, int bodies, int timestep, int panel_size, int TILELENGTH){
	__shared__ double tys[7];

    int body = blockIdx.x*TILELENGTH + threadIdx.x;
    
    if (body < bodies){
    	rk4(ys+timestep*panel_size+body*7, tys, h, body, ys+timestep*panel_size, bodies);
		for (int j = 0; j < 7; j++){
			// assign those values to the state array
			ys[(timestep)*panel_size+body*7+j] = tys[j];
		}
	}
}



__device__ void rk4(double* y0, double* result, float h, int ID, double* pList, int bodies){
	double k1[7];
	double k2[7];
	double k3[7];
	double k4[7];

	double y1[7];
	double y2[7];
	double y3[7];

	model(k1, y0, ID, pList, bodies);
	vecMult(k1, k1, h, 7);
	vecDiv(y1, k1, 2, 7);
	elementWiseAdd(y1, y0, y1, 7); 

    model(k2, y1, ID, pList, bodies);
	vecMult(k2, k2, h, 7);
	vecDiv(y2, k2, 2, 7);
	elementWiseAdd(y2, y0, y2, 7); 

    model(k3, y2, ID, pList, bodies);
	vecMult(k3, k3, h, 7);
	elementWiseAdd(y3, y0, k3, 7); 

    model(k4, y3, ID, pList, bodies);
	vecMult(k4, k4, h, 7);

	vecDiv(y1, k1, 6, 7);
	elementWiseAdd(result, y0, y1, 7);

	vecDiv(y1, k2, 3, 7);
	elementWiseAdd(result, result, y1, 7);

	vecDiv(y1, k3, 3, 7);
	elementWiseAdd(result, result, y1, 7);

	vecDiv(y1, k4, 6, 7);
	elementWiseAdd(result, result, y1, 7);

}
