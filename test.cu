#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define pathSize 31		// dataset size in number of coordinates
#define popSize 320		// population size
#define ThreadNum 50	// ThreadNum * BlockNum must be equal to popSize -> when possible, the granularity reaches the chromosome level -> we need enough threads
#define BlockNum 2
#define subPopSize 32	// popSize must be a multiple of this and this should be a multiple of the warp size (32)

// A general debug error for cuda
void checkCUDAError(const char *msg = NULL) {
	cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
		printf("Error after call to %s\n", msg);
        // Additional error handling if needed
    }
}


// Device function to shuffle the chromosomes in the given portion of the population
__device__ void device_random_shuffle(int *chromosome, int thread){
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);	// init random seed
	int gene;
	for(int i=0;i<pathSize;i++){
		int idx1 = int(curand_uniform(&state)*pathSize);
		int idx2 = int(curand_uniform(&state)*pathSize);	// random from 0 to 1 * path size
		gene = chromosome[idx1];
		chromosome[idx1] = chromosome[idx2];
		chromosome[idx2] = gene;
	}
}

// A kernel to shuffle the chromosomes in the population
__global__ void random_shuffle(int *population){
	int thread = blockIdx.x * subPopSize + threadIdx.x;  // thread index
	int index = thread * pathSize;    // index of the subpopulation of chromosomes considering stride
	device_random_shuffle(population+index, thread);
}


__device__ double device_calculate_distance(int *chromosome, double *distance_matrix){
	double distance = 0.0;
	for (int i = 0; i < pathSize-1; i++){
		distance += distance_matrix[chromosome[i]*pathSize + chromosome[i+1]];
	}

	distance += distance_matrix[chromosome[pathSize-1]*pathSize + chromosome[0]];
	return distance;
}

// Calculate Scores and fill the distance and fitness arrays for every individual
__global__ void calculate_scores(int *population, double *distance_matrix,  double *population_fitness, double *population_distances){
	int thread = blockIdx.x * subPopSize + threadIdx.x;
	int index = thread * pathSize;
	population_distances[thread] = device_calculate_distance(population+index, distance_matrix);
	population_fitness[thread] = 10000 / population_distances[thread];
	// note: the thread serves as the index of the single individual in the population in the fitness and distance arrays
}

// Swap two chromosomes
__global__ void parallel_swap(int *population, int i, int j){
	int index = threadIdx.x;
	int temp = population[i*pathSize+index];
	population[i*pathSize+index] = population[j*pathSize+index];
	population[j*pathSize+index] = temp;
}

__device__ void device_fit_sort_subpop(int* sub_population, double* sub_population_fitness){
	// other algorithms could be employed to narrow the granularity but bubblesort should do it's job on a small enough subpopulation
	// here a consideration must be done on the structure of data:
	// sub_population: 			[chromosome_1 : [0,1,2,3,4,...,30] , chromosome_2 : [0,1,...,30]] 	subPopSize*pathSize integers
	// sub_population_fitness : [fitness_1, fitness_2, ... , subPopSize]						subPopSize doubles
	// sort the subpopulation based on the fitness
	for (int i = 0; i < subPopSize; i++){
		for (int j = 0; j < subPopSize-1; j++){
			if (sub_population_fitness[j] < sub_population_fitness[j+1]){
				// swap the fitness
				double temp = sub_population_fitness[j];
				sub_population_fitness[j] = sub_population_fitness[j+1];
				sub_population_fitness[j+1] = temp;
				// swap all the genes in the chromosome sequentially on the thread
				
				for (int k = 0; k < pathSize; k++){
					int temp = sub_population[j*pathSize+k];
					sub_population[j*pathSize+k] = sub_population[(j+1)*pathSize+k];
					sub_population[(j+1)*pathSize+k] = temp;
				}
				/*
				// swap all the genes in the chromosome in parallel on pathSize threads
				parallel_swap<<<1,pathSize>>>(sub_population, j, j+1);
				*/
			}
		}
	}

}

// Sort a subpopulation on the granularity of subpopulation
__global__ void fit_sort_subpop(int *population, double *population_fitness){
	int fitness_index = threadIdx.x * subPopSize;	// index of the first fitness of each subpopulation
	int sub_pop_index = threadIdx.x * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
	device_fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
}

// Helper function to load data from file
void load_data(int *coordinates, char *filename){
	// read filename
	FILE *file = fopen(filename, "r");
	if (file == NULL){
		printf("Error opening file %s\n", filename);
		exit(1);
	}
	int i = 0;
	while (fscanf(file, "%d %d", &coordinates[i*2], &coordinates[i*2+1]) != EOF){
		i++;
	}
}

// IMPORTANT:
// - The alignment of data is guaranteed by the local operations in a parallel context, that is,
//  f.i. if a a crossover operator removes N/2 elements from the population, the remaining elements will be aligned in the same way as the original ones
//  all the other parallel operations will be aligned in the same way - in this sense, all the STEPS in a GA are essentially sequential.
//  Parallelism is used to speed up the operations, not to change the order of the operations themselves.
int main(){
														/*SERIAL PART OF THE CODE*/

	//-------------------------------------------------
	// Load the coordinates of the cities from file
	//-------------------------------------------------
    int *path_coordinates = (int*)malloc(pathSize * 2 * sizeof(int));	//[pathSize][2];
    load_data(path_coordinates, "data_1.txt");

	// as a test print the coordinates
	// for (int i=0;i<pathSize;i++){
	// 	printf("%d %d\n",path_coordinates[i*2],path_coordinates[i*2+1]);
	// }


	//-----------------------------------------
	// Allocate and fill the distance matrix
	//-----------------------------------------
	double *distance_matrix = (double*)malloc(pathSize*pathSize*sizeof(double));	//[pathSize][pathSize];
	for(int i = 0; i < pathSize; i++){
		for(int j = 0; j < pathSize; j++){
			distance_matrix[i*pathSize+j] = sqrt(pow(path_coordinates[i*2]-path_coordinates[j*2],2) + pow(path_coordinates[i*2+1]-path_coordinates[j*2+1],2));
		}
	}

	// as a test print the distance matrix
	// for (int i=0;i<pathSize;i++){
	// 	for (int j=0;j<pathSize;j++){
	// 		printf("%d\t ",(int)distance_matrix[i*pathSize+j]);
	// 	}
	// 	printf("\n");
	// }

	//---------------------------------------------------------------
	// Allocate and fill the population in RAM
	//---------------------------------------------------------------
	int *population = (int*)malloc(popSize * pathSize * sizeof(int));	//[popSize][pathSize]; - This represents the order of the cities for each chromosome in the population
	//GPU::create pathSize sequences aka chromosomes - each chromosome is composed of a sequence of indices representing the cities
	for (int i = 0; i < pathSize * popSize; i++){
		population[i] = i % pathSize;
		
	}

	// instantiate the gpu_population in the VRAM
	int *gpu_population;
	cudaMalloc(&gpu_population, popSize * pathSize * sizeof(int));
	cudaMemcpy(gpu_population, population, popSize * pathSize * sizeof(int), cudaMemcpyHostToDevice);

	//---------------------------------------------------------------
	// Allocate fitness and distances for each individual
	//---------------------------------------------------------------
	double *population_fitness = (double*)malloc(popSize*sizeof(double));	//[popSize];
	double *population_distances = (double*)malloc(popSize*sizeof(double));	//[popSize];
	// these are needed in RAM for the final selection but are used in the GPU for the rest of the operations

	// different granularity levels for grids and block
	dim3 c_grid(popSize/subPopSize,1,1);	dim3 c_block(subPopSize,1,1);					// chromosome granularity
	dim3 g_grid(popSize/subPopSize,1,1);	dim3 g_block(pathSize,subPopSize,1);			// gene granularity
	dim3 s_grid(1,1,1);						dim3 s_block(popSize/subPopSize,1,1);			// subpopulation granularity

														/*PARALLEL PART OF THE CODE*/

	//---------------------------------------------------------------
	// Random shuffle the population in the GPU for the first time
	//---------------------------------------------------------------
	    
    // execute a random shuffle of the population
	random_shuffle<<<c_grid,c_block>>>(gpu_population);					// chromosome granularity
    // Check for any errors launching the kernel
	checkCUDAError("random_init_shuffle kernel launch");

    cudaDeviceSynchronize();				// IS THIS NECESSARY??

	//-----------------------------------------------------------------------------------
	// Allocate fitness and distances for each individual on the GPU and calculate them
	//-----------------------------------------------------------------------------------
	double *gpu_distance_matrix;
	cudaMalloc(&gpu_distance_matrix, pathSize*pathSize*sizeof(double));
	cudaMemcpy(gpu_distance_matrix, distance_matrix, pathSize*pathSize*sizeof(double), cudaMemcpyHostToDevice);
	double *gpu_population_fitness;		// for now just an empty array
	cudaMalloc(&gpu_population_fitness, popSize*sizeof(double));
	double *gpu_population_distances;	// for now just an empty array
	cudaMalloc(&gpu_population_distances, popSize*sizeof(double));

	// execute the fitness + distance kernel	(fitness can be defined as the inverse of the distance or equivalently as alpha/distance where alpha is a constant)
	calculate_scores<<<c_grid,c_block>>>(gpu_population, gpu_distance_matrix, gpu_population_fitness, gpu_population_distances);
	// Check for any errors launching the kernel
	checkCUDAError("calculate_scores kernel launch");

	cudaDeviceSynchronize();				// IS THIS NECESSARY??

	// as a test print the fitness and distances
	cudaMemcpy(population_fitness, gpu_population_fitness, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(population_distances, gpu_population_distances, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	for (int i=0;i<popSize;i++){
		printf("Fitness: %f\tDistance: %f\n",population_fitness[i],population_distances[i]);
	}


	//------------------------------------------------------------------------------------------------------------------------------
	// Start the actual GA -- Note: GAs are intrinsically sequential: selection -> crossover -> mutation -> (migration) -> next gen
	// in practice the parallelization takes pace within single steps
	//------------------------------------------------------------------------------------------------------------------------------

	// THE FOLLOWING ASSIGNMENT SHOULD BE DETERMINED A POSTERIORI OF AN INSPECTION OF THE DEVICE BASED ON CERTAIN CRITERIONS...
	while(true /*stop criterion*/){
		// Sort each sub-population - this can only be done on the granularity of subpopulation, hence it's outside of the genetic step GA
		fit_sort_subpop<<<s_grid,s_block>>>(gpu_population, gpu_population_fitness);
		// Check for any errors launching the kernel
		checkCUDAError("fit_sort_subpop kernel launch");
		break;
	}
	cudaDeviceSynchronize();
	// Copy back to RAM the gpu_population onto the population pointer
	cudaMemcpy(population, gpu_population, pathSize*popSize*sizeof(int), cudaMemcpyDeviceToHost);
	// copy fitness and distances
	cudaMemcpy(population_fitness, gpu_population_fitness, popSize*sizeof(double), cudaMemcpyDeviceToHost);

	// as a test print a sub-population with fitness values for each chromosome
	for (int f=0;f<subPopSize;f++){
		printf("Fitness: %f\t", population_fitness[f]);
		for (int i=0;i<pathSize;i++){
			printf("%d ",population[f*pathSize+i]);
			if((i+1)%pathSize==0){
				printf("\n");
			}
		}
	}
	


	// As a test print the shuffled population
    // Copy back to RAM the gpu_population onto the population pointer
	// cudaMemcpy(population,gpu_population,pathSize*popSize*sizeof(int),cudaMemcpyDeviceToHost);
	// as a test print population
	// for (int i=0;i<popSize*pathSize;i++){
	// 	printf("%d ",population[i]);
	// 	if((i+1)%pathSize==0){
	// 		printf("\n");
	// 	}
	// }



	return 0;
}