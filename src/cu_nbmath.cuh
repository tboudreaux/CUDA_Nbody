#ifndef NBMATH
#define NBMATH

/* Function -> vecAdd: Add a scaler to a vector
 *
 * Params:
 * 		 result -> array to store the resultant vector A +. B    [double *]
 * 		 A      -> array to add B to by operation A +. B         [double *]
 * 		 B      -> scaler to add to array B by operation A +. B  [double]
 * 		 sizeA  -> size of array A                               [int]
 *
 * Preconditions:
 *       - A allocated and given values
 *       - result allocated to be of sizeA doubles
 *
 * Postconditions:
 * 		 - results filled with results of operation A +. B
 *
 */
__device__ void vecAdd(double* result, double* A, double B, int sizeA);

/* Function -> vecMult: Multiply a scaler to a vector
 *
 * Params:
 * 		 result -> array to store the resultant vector A *. B    [double *]
 * 		 A      -> array to add B to by operation A *. B         [double *]
 * 		 B      -> scaler to add to array B by operation A *. B  [double]
 * 		 sizeA  -> size of array A                               [int]
 *
 * Preconditions:
 *       - A allocated and given values
 *       - result allocated to be of sizeA doubles
 *
 * Postconditions:
 * 		 - results filled with results of operation A *. B
 *
 */
__device__ void vecMult(double* result, double* A, double B, int sizeA);

/* Function -> vecDiv: Divide a scaler to a vector
 *
 * Params:
 * 		 result -> array to store the resultant vector A /. B    [double *]
 * 		 A      -> array to add B to by operation A /. B         [double *]
 * 		 B      -> scaler to add to array B by operation A /. B  [double]
 * 		 sizeA  -> size of array A                               [int]
 *
 * Preconditions:
 *       - A allocated and given values
 *       - result allocated to be of sizeA doubles
 *
 * Postconditions:
 * 		 - results filled with results of operation A /. B
 *
 */
__device__ void vecDiv(double* result, double* A, double B, int sizeA);

/* Function -> elementWiseAdd: Add a two vectors element by element
 *
 * Params:
 * 		 result -> array to store the resultant vector A + B    [double *]
 * 		 A      -> array to add B to by operation A + B         [double *]
 * 		 B      -> array to add to array B by operation A + B   [double *]
 * 		 sizeA  -> size of arrays A&B                           [int]
 *
 * Preconditions:
 *       - A allocated and given values
 *       - result allocated to be of sizeA doubles
 *
 * Postconditions:
 * 		 - results filled with results of operation A + B
 *
 */
__device__ void elementWiseAdd(double* result, double* A, double* B, int size);


#endif
