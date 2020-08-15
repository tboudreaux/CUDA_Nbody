#ifndef CPHYS
#define CPHYS


/* Function -> model: differential model to be integrated
 * 	  
 * Params:
 * 		  dydt   -> array to store derivitives in                     [double *]
 *	      y0     -> Initial Conditions array                          [double *]
 *		  ID     -> ID of particle being integrated                   [int]
 *		  pList  -> List of all partices in simulation at time step t [double *]
 *		  bodies -> Total number of particles in simulation           [int]
 *
 * Precondition:
 * 		  - y0 allocated and filled with initial conditions           [7 doubles]
 * 		  - dydt allocated                                            [7 doubles]
 * 		  - pList allocated and filled with particle data             [7*bodies doubles]
 * Postcondition:
 * 		  - dydt filled with derivitives
 */
__device__ void model(double* dydt, double* y0, int ID, double* pList, int bodies);



/* Function -> rk4: Runge-Kutte 4th Order integrator
 *
 * Params:
 *        y0     -> initial conditions array for integration           [double *]
 *        result -> array to place results of integration into         [double *]
 *		  h      -> time step to use when integrating                  [float]
 *	      ID     -> ID of particle being integrated                    [int]
 *		  pList  -> List of all particles in simulation at time step t [double *]
 *		  bodies -> Total number of particles in simulation            [int]
 *
 * Precondition:
 * 		  - y0 allocated and filled with initial conditions            [7 doubles]
 *		  - result allocated 										   [7 boubles]
 * 		  - pList allocated and filled with particle data              [7*bodies doubles]
 *
 * Postcondition:
 *        - result filled with the conditions for the next time step
 *
 */
__device__ void rk4(double* y0, double* result, float h, int ID, double* pList, int bodies);


/* Function -> time_step: step the simulation through one timestep, using the gpu
 *                        to integrate all bodies at once
 *
 * Params:
 *         ys         -> complete state of all the particles at the outset of the time step      [double *]
 *         h          -> timestep size to step all particles through                             [float]
 *         bodies     -> total number of bodies in simulation                                    [int]
 *         timestep   -> the current time step that the simulation is on                         [int] 
 *         panel_size -> the size, in number of doubles of information per time step (7*bodies)  [int]
 *         TILELENGTH -> the size of the tile to use in the kernel code
 * 
 * Preconditions:
 *         - ys allocated on the GPU, initial state loaded, prior time steps also loaded
 * Postcondition:
 *         - of the total size of ys (7*bodies*timesteps) a further 7*bodies (one timestep)
 *           will be loaded into ys. it will imediatly proceed the previous timestep in mem
 */
__global__ void time_step(double* ys, float h, int bodies, int timestep, int panel_size, int TILELENGTH);


#endif
