#include <iostream>
#include<cstdlib>

#include "cu_nbmath.cuh"

using namespace std;

__device__ void vecAdd(double* result, double* A, double B, int sizeA){
	for (int i = 0; i < sizeA; i++){
		result[i] = A[i]+B;
	}
}

__device__ void vecMult(double* result, double* A, double B, int sizeA){
	for (int i = 0; i < sizeA; i++){
		result[i] = A[i]*B;
	}
}

__device__ void vecDiv(double* result, double* A, double B, int sizeA){
	for (int i = 0; i < sizeA; i++){
		result[i] = A[i]/B;
	}
}

__device__ void elementWiseAdd(double* result, double* A, double* B, int size){
	for (int i = 0; i < size; i++){
		result[i] = A[i]+B[i];
	}
}

