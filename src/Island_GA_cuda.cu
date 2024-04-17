#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define generations 100		// number of generations

#define threadsPerBlock 8		// number of threads per block

#define pathSize 48				// dataset size in number of coordinates
#define popSize 640000				// population size
#define subPopSize 32				// popSize must be a multiple of this and this should be a multiple of the warp size (32)
#define selectionThreshold 0.7		// the threshold (%) for the selection of the best chromosomes in the sub-populations
#define migrationAttemptDelay 10	// the number of generations before a migration attempt is made
#define migrationProbability 0.7	// the probability that a migration happens between two islands
#define migrationNumber 4			// the number of chromosomes that migrate between two islands - is supposed to be less than a half of subPopSize

// CROSSOVER CONTROL PARAMETERS
#define alpha 0.2	// environmental advantage
#define beta 0.7	// base hospitality
// The probability that a crossover happens on a certain island is in the range [alpha/(popSize/subPopSize) + beta , alpha + beta] - a minimum of beta is granted for each island

// MUTATION CONTROL PARAMETERS
#define gamma 0.2	// environmental replication disadvantage
#define delta 0.3	// base replication disadvantage
// The probability that a mutation happens on a certain island is in the range [gamma/(popSize/subPopSize) + delta , gamma + delta] - a minimum of delta is granted for each island

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

// Calculate the distance of a chromosome - TSP solution
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

// Selection of the best chromosomes in the subpopulations - in practice, to emulate the fact that only the fittest get a chance to reproduce, who is less than selectionThreshold% of the best is overwritten
__device__ void device_selection(int *sub_population, double *sub_population_fitness){
	// substitute the last selectionThreshold% of the population with the first part of the population, like: [11,8,7,5,4,3,2,1,0,-1] t = 30% -> [11,8,7,5,4,3,2,7,8,11]
	// define first selectionThreshold% of the population
	int threshold = (int)(subPopSize * selectionThreshold);
	for (int i = 0 ; i < threshold; i++){
		sub_population_fitness[subPopSize-i] = sub_population_fitness[i];
		for(int j = 0; j < pathSize; j++){
			sub_population[i*pathSize+j] = sub_population[(subPopSize-i)*pathSize+j];
		}
	}
}

// crossover of two chromosomes
__device__ void device_distance_crossover(int *parent1, int *parent2, int *offspring, double *distance_matrix){
	// init the random seed
	curandState state;
	curand_init((unsigned long long)clock()+blockIdx.x,0,0,&state);	// init random seed
	// select a random point in parent 1
	int first = int(curand_uniform(&state)*pathSize);
	// use it as first part of the offspring
	offspring[0] = parent1[first];
	// starting from the first point in both parents, compare the distance with the previous point in the offspring and choose the one with the smallest distance if it is not already in the offspring
	// if both are already in the offspring, choose a random one not in the offspring
	int offspring_size = 1;
	int i = 0;
	while (offspring_size < pathSize){
		int last = offspring[offspring_size-1];
		int next1 = parent1[i];
		int next2 = parent2[i];
		bool in_offspring1 = false;
		bool in_offspring2 = false;
		for (int j = 0; j < offspring_size; j++){
			if (offspring[j] == next1){
				in_offspring1 = true;
			}
			if (offspring[j] == next2){
				in_offspring2 = true;
			}
			if (in_offspring1 && in_offspring2){
				break;
			}
		}
		if (in_offspring1 && in_offspring2){
			// choose a random one not in the offspring
			int next = parent1[int(curand_uniform(&state)*pathSize)];
			bool in_offspring = true;
			while (in_offspring){
				in_offspring = false;
				for (int j = 0; j < offspring_size; j++){
					if (offspring[j] == next){
						next = parent1[int(curand_uniform(&state)*pathSize)];
						in_offspring = true;
						break;
					}
				}
			}
			offspring[offspring_size] = next;
			offspring_size++;
			i++;
			continue;
		}else if (in_offspring1){
			offspring[offspring_size] = next2;
			offspring_size++;
		}else if (in_offspring2){
			offspring[offspring_size] = next1;
			offspring_size++;
		}else{
			// if both are not in the offspring, choose the one with the smallest distance
			double d1 = distance_matrix[last*pathSize+next1];
			double d2 = distance_matrix[last*pathSize+next2];
			if (d1 <= d2){
				offspring[offspring_size] = next1;
				offspring_size++;
			}else{
				offspring[offspring_size] = next2;
				offspring_size++;
			}
		}
		i++;
	}
}

__device__ void device_crossover(int *sub_population, double *distance_matrix){
	// for this subpopulation define the crossover rate
	double crossover_rate = (alpha * (blockIdx.x+1))/(popSize/subPopSize) + beta;
	
	// init the random seed
	curandState state;
	curand_init((unsigned long long)clock()+blockIdx.x,0,0,&state);	// init random seed

	// starting from the top down, each pair of chromosomes i,i+1 is crossed and the offspring takes the place of the last chromosome in the subpopulation in ascending order
	// Fittest Roulette!
	int new_borns = 0;
	int i = 0;
	int j = subPopSize-1;
	int remaining_top_percent = int(subPopSize*selectionThreshold);
	// if odd number of selected chromosomes, increase the remaining top percent to let mate the best of the excluded
	if (int(subPopSize*selectionThreshold) % 2 == 1){
		remaining_top_percent += 1;
	}
	int excluded_chromosomes = subPopSize - remaining_top_percent;

	while (new_borns < subPopSize-1){
		//printf("thread: %d, i: %d, j: %d, new_borns: %d, remaining_top_percent: %d, excluded_chromosomes: %d\n", threadIdx.x, i, j, new_borns, remaining_top_percent, excluded_chromosomes);
		// if all the non-mating chromosomes are already overwritten, decrease the remaining top percent to overwrite the last mating chromosome
		if (new_borns >= excluded_chromosomes){
			if (excluded_chromosomes == subPopSize-2){
				i = 0;
			}
			else{
				remaining_top_percent -= 2;
				excluded_chromosomes += 2;
			}
		}
		i = i % remaining_top_percent;	// wrap around the remaining top percent

		if (curand_uniform(&state) < crossover_rate){
			// crossover the two chromosomes
			device_distance_crossover(sub_population+i*pathSize, sub_population+(i+1)*pathSize, sub_population+j*pathSize, distance_matrix);
			new_borns += 1;
			j -= 1;
		}
		i += 2;	// move to the next pair of chromosomes
	}
	//printf("Crossover completed on thread: %d\n", blockIdx.x * threadsPerBlock + threadIdx.x);
}

__global__ void genetic_step(int *population, double *population_fitness, double *distance_matrix){
	int fitness_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	int sub_pop_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
	if (fitness_index < popSize){
		// Sort the subpopulation
		device_fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
		// Selection is implicit with the sorting
		// Crossover
		device_crossover(population+sub_pop_index, distance_matrix);
		//printf("Genetic step completed on thread: %d\n", blockIdx.x * threadsPerBlock + threadIdx.x);
	}
}


__global__ void mutation(int *population){
	double mutation_rate = (gamma * (blockIdx.x+1))/(popSize/subPopSize) + delta;
	int thread = blockIdx.x * subPopSize + threadIdx.x;
	// except the case the thread is the first of the subpopulation, mutate with a certain probability
	if (thread % subPopSize != 0){
		curandState state;
		curand_init((unsigned long long)clock()+thread,0,0,&state);	// init random seed
		if (curand_uniform(&state) < mutation_rate){
			// mutate the chromosome swapping two random genes
			int idx1 = int(curand_uniform(&state)*pathSize);// random from 0 to 1 * path size
			int idx2 = (idx1 + 1) % pathSize;
			int temp = population[thread*pathSize+idx1];
			population[thread*pathSize+idx1] = population[thread*pathSize+idx2];
			population[thread*pathSize+idx2] = temp;
		}
	}
}

__global__ void migration(int *population){
	// Migrate the first migrationNumber chromosomes from subpopulation i to subpopulation i+1 in a ring fashion
	//int sub_pop_index = threadIdx.x;
	int sub_pop_index = blockIdx.x * threadsPerBlock + threadIdx.x;
	if (sub_pop_index >= popSize/subPopSize){
		return;
	}
	//printf("\nThread: %d", blockIdx.x * threadsPerBlock + threadIdx.x);
	int next_sub_pop_index = (sub_pop_index+1) % (popSize/subPopSize);
	//printf("Attempting to migrate from %d to %d\n", sub_pop_index, next_sub_pop_index);
	for (int i = 0; i < migrationNumber; i++){
		// swap the chromosome i from subpopulation sub_pop_index with the chromosome subPopSize-migrationNumber+i from subpopulation next_sub_pop_index
		for (int j = 0; j < pathSize; j++){
			int temp = population[sub_pop_index*subPopSize*pathSize+i*pathSize+j];
			population[sub_pop_index*subPopSize*pathSize+i*pathSize+j] = population[next_sub_pop_index*subPopSize*pathSize+(subPopSize-migrationNumber+i)*pathSize+j];
			population[next_sub_pop_index*subPopSize*pathSize+(subPopSize-migrationNumber+i)*pathSize+j] = temp;
		}
	}
	//printf("Migration from %d to %d completed\n", sub_pop_index, next_sub_pop_index);
	// NOTE: we suppose that migrationNumber is always less than half of the subpopulation size to avoid write before read conflicts
}

// Sort the subpopulations based on the fitness
__global__ void fit_sort(int *population, double *population_fitness){
	//int fitness_index = threadIdx.x * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	//int sub_pop_index = threadIdx.x * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
	int fitness_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	int sub_pop_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize * pathSize;	// index of the first chromosome of each subpopulation

	if (fitness_index >= popSize){
		return;
	}
	// Sort the subpopulation
	device_fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
}

// Helper function to load data from file
void load_data(int *coordinates, const char *filename){
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

void save_best_solution(int *best_chromosome, int *coordinates){
	FILE *file = fopen("../results/best_solution.txt", "w");
	if (file == NULL){
		printf("Error opening file best_solution.txt\n");
		exit(1);
	}
	for (int i = 0; i < pathSize; i++){
		fprintf(file, "%d %d\n", coordinates[best_chromosome[i]*2], coordinates[best_chromosome[i]*2+1]);
	}
	fclose(file);
}

// IMPORTANT:
// - The alignment of data is guaranteed by the local operations in a parallel context, that is,
//  f.i. if a a crossover operator removes N/2 elements from the population, the remaining elements will be aligned in the same way as the original ones
//  all the other parallel operations will be aligned in the same way - in this sense, all the STEPS in a GA are essentially sequential.
//  Parallelism is used to speed up the operations, not to change the order of the operations themselves.
int main(){
														/*SERIAL PART OF THE CODE*/
	clock_t start = clock();

	//-------------------------------------------------
	// Load the coordinates of the cities from file
	//-------------------------------------------------
    int *path_coordinates = (int*)malloc(pathSize * 2 * sizeof(int));	//[pathSize][2];
    load_data(path_coordinates, "../data/48_cities.txt");
	//-----------------------------------------
	// Allocate and fill the distance matrix
	//-----------------------------------------
	double *distance_matrix = (double*)malloc(pathSize*pathSize*sizeof(double));	//[pathSize][pathSize];
	for(int i = 0; i < pathSize; i++){
		for(int j = 0; j < pathSize; j++){
			distance_matrix[i*pathSize+j] = sqrt(pow(path_coordinates[i*2]-path_coordinates[j*2],2) + pow(path_coordinates[i*2+1]-path_coordinates[j*2+1],2));
		}
	}
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
	//dim3 s_grid_alternative(popSize/subPopSize,1,1);	dim3 s_block_alternative(1,1,1);
	dim3 s_grid(ceil((popSize/subPopSize) * (1.0/threadsPerBlock)),1,1); dim3 s_block(threadsPerBlock,1,1); // subpopulation granularity

	printf("Using %d total threads in subpopulation granularity\n", s_grid.x*s_block.x);
	printf("Of which %d blocks containing %d threads each\n", s_grid.x, s_block.x);

	// dim3 s_grid(ceil((popSize/subPopSize) * (1.0/32.0)),1,1) dim3 s_block(32,1,1)	// subpopulation granularity

	clock_t load_checkpoint = clock();
	fprintf(stderr,"Data loaded in %.2f ms\n", ((double) (load_checkpoint - start)) * 1000.0 / CLOCKS_PER_SEC);

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
	// cudaMemcpy(population_fitness, gpu_population_fitness, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(population_distances, gpu_population_distances, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	// for (int i=0;i<popSize;i++){
	// 	printf("Fitness: %f\tDistance: %f\n",population_fitness[i],population_distances[i]);
	// }


	//------------------------------------------------------------------------------------------------------------------------------
	// Start the actual GA -- Note: GAs are intrinsically sequential: selection -> crossover -> mutation -> (migration) -> next gen
	// in practice the parallelization takes pace within single steps
	//------------------------------------------------------------------------------------------------------------------------------
	int generation = 1;
	srand(time(NULL));	// seed the random number generator
	printf("Starting the GA\n");
	while(generation <= generations){
		clock_t start_gen = clock();

		// Execute the GA steps
		printf("Starting Genetic step\n");
		genetic_step<<<s_grid,s_block>>>(gpu_population, gpu_population_fitness, gpu_distance_matrix); // subpopulation granularity
			checkCUDAError("genetic_step kernel launch");
			cudaDeviceSynchronize();				// IS THIS NECESSARY??
		
		printf("Starting Mutation\n");
		mutation<<<c_grid,c_block>>>(gpu_population);	// chromosome granularity		IS IT CONVENIENT?
			checkCUDAError("mutation kernel launch");
			cudaDeviceSynchronize();				// IS THIS NECESSARY??
		// calculate the fitness and distances for the new generation
		if (generation % migrationAttemptDelay == 0 && rand()/RAND_MAX < migrationProbability){
			// Migration - we suppose when a single island is able to migrate it forces the other islands to migrate as well
			printf("Starting Migration\n");
			migration<<<s_grid,s_block>>>(gpu_population); // subpopulation granularity
				checkCUDAError("migration kernel launch");
				cudaDeviceSynchronize();				// IS THIS NECESSARY??
		}
		// realign the population fitness and distances for the new generation
		printf("Starting Fitness and Distance Calculation\n");
		calculate_scores<<<c_grid,c_block>>>(gpu_population, gpu_distance_matrix, gpu_population_fitness, gpu_population_distances);
			checkCUDAError("calculate_scores kernel launch");
			cudaDeviceSynchronize();				// IS THIS NECESSARY??
		//printf("Generation: %d\n", generation);
		generation++;

		clock_t end_gen = clock();
		fprintf(stderr,"Generation %d completed in %.2f ms\n", generation-1, ((double) (end_gen - start_gen)) * 1000.0 / CLOCKS_PER_SEC);
	}
	// Sort the final population one more time
	fit_sort<<<s_grid,s_block>>>(gpu_population, gpu_population_fitness); // subpopulation granularity
	cudaDeviceSynchronize();				// IS THIS NECESSARY??

	// move back to host memory the final population
	cudaMemcpy(population, gpu_population, popSize * pathSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(population_fitness, gpu_population_fitness, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	// considering each island (subpopulation) are sorted, select the best amog all the islands
	double best_fitness = 0.0;
	int best_index = 0;
	for (int i = 0; i < popSize; i += subPopSize){
		if (population_fitness[i] > best_fitness){
			best_fitness = population_fitness[i];
			best_index = i;
		}
	}
	// extract the best chromosome from the population
	int *best_chromosome = (int*)malloc(pathSize*sizeof(int));
	for (int i = 0; i < pathSize; i++){
		best_chromosome[i] = population[best_index*pathSize+i];
	}

	// also use the best_chromosome[i] to index the cities from the coordinates and append to a file
	save_best_solution(best_chromosome, path_coordinates);

	// calculate the distance of the best chromosome
	double best_distance = 0.0;
	for (int i = 0; i < pathSize-1; i++){
		best_distance += distance_matrix[best_chromosome[i]*pathSize + best_chromosome[i+1]];
	}
	best_distance += distance_matrix[best_chromosome[pathSize-1]*pathSize + best_chromosome[0]];

	// print the best chromosome and the distance
	printf("Best chromosome: ");
	for (int i = 0; i < pathSize; i++){
		printf("%d ", best_chromosome[i]);
	}
	printf("\nBest distance: %f\n", best_distance);

	// free the memory
	free(path_coordinates);
	free(distance_matrix);
	free(population);
	free(population_fitness);
	free(population_distances);
	free(best_chromosome);
	cudaFree(gpu_population);
	cudaFree(gpu_distance_matrix);
	cudaFree(gpu_population_fitness);
	cudaFree(gpu_population_distances);

	clock_t end = clock();
	fprintf(stderr,"Execution completed in %.2f ms\n", ((double) (end - start)) * 1000.0 / CLOCKS_PER_SEC);

	// exit
	return 0;
}