#ifndef UTILS
#define UTILS

#include <string>

double* load(const std::string& filename, int &size);

/* Function -> print_state: print a single time step of the
 * 	  
 * Params:
 * 		  state    -> state of model at some time t                   [double *]
 *	      bodies   -> total number of bodies in model                 [int]
 *
 * Precondition:
 * 	      - state is allocated and filled with data at time t         [7*bodies doubles]
 * 	        on all bodies in system 
 *
 * Postcondition:
 * 		  - state is written to standard output
 */
void print_state(double* state, int bodies);

/* Function -> dumpState: write the entire simulation results at each time step
 * 			   t to a file on disk
 * 	  
 * Params:
 * 		  filename    -> filename to write simulation data to            [const string&]
 *	      state       -> state of each particle at each time step        [double *]
 *	      timesteps   -> total number of timesteps system integrated for [long int]
 *	      bodies      -> total number of bodies in model                 [int]
 *
 * Precondition:
 * 	      - state is allocated and filled with data at all times t       [7*bodies doubles]
 * 	        on all bodies in system 
 *
 * Postcondition:
 * 		  - state at all times t is written to file on disk
 * 		  	- State info written as 1D binary array
 * 		  	- ascii line precedes with bodies, timesteps\n
 */
void dumpState(const std::string& filename, double* state, long int timesteps, int bodies);

/* Function -> print_seven_vector: print out a 7 component vector to standard output
 * 	  
 * Params:
 * 		  vec         -> vector to print out                          [double *]
 *
 * Precondition:
 * 	      - vec is allocated and filled with 7 doubles                [7]
 *
 * Postcondition:
 * 		  - vec is written to standard output
 */
void print_seven_vector(double* vec);

/* Function -> arange: create an array from low to high with steps of step
 *
 * Params:
 * 		 ts     -> pointer to space where results of arange will live    [double *]
 * 		 low    -> low bound for arange array                            [float]
 * 		 high   -> high bound for arange array                           [float]
 * 		 step   -> step size between values in arange array              [float]
 * 		 size   -> number of elements in arange array                    [&int]
 *
 * Preconditions:
 *       - ts pointer declated [not allocated]
 *
 * Postconditions:
 * 		 - ts pointer allocated and filled with arange array
 * 		 - size filled with the size of the arange array
 */
void arange(float low, float high, float step, long int &size);


/* Function -> errorCheck: check for, and handel errors that CUDA throghs
 *
 * Params:
 *        code   -> Eror Identification code                           [int]
 *		  err    -> cude error thrown                                  [cudaError_t]
 *
 * Precondition:
 * 		  - valid cudaError_t caught from cuda function
 *
 * Postcondition:
 *        - If a CUDA error has been raised the program will exit with EXIT_FAILURE and 
 *          display that error to standard error.
 *
 */
void errorCheck(int code, cudaError_t err);

/* Function -> int_n_model: integration controller over some time t
 *
 * Params:
 *        y0        -> initial conditions array for integration           [double *]
 *        t         -> total time to integrate over                       [double]
 *        h         -> time step to use when integrating                  [float]
 *        bodies    -> Total number of particles in simulation            [int]
 *        timesteps -> total number of timesteps, to be filled in func    [&int]
 *
 * Precondition:
 *        - y0 allocated and filled with initial conditions               [7 doubles]
 *
 * Postcondition:
 *        - Function integrated over time t with time steps h, all time states
 *          returned in double* array by function.
 *
 */
double* int_n_model(double* y0, double t, float h, int bodies, long int &timesteps);

#endif
