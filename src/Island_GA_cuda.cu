#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <unistd.h>

#define generations 50		// number of generations

#define threadsPerBlock 64		// number of threads per block

#define pathSize 48					// dataset size in number of coordinates
#define popSize 1024					// population size
#define subPopSize 32				// popSize must be a multiple of this and this should be a multiple of the warp size (32)
#define selectionThreshold 0.5		// the threshold (%) for the selection of the best chromosomes in the sub-populations
#define migrationAttemptDelay 10	// the number of generations before a migration attempt is made
#define migrationProbability 0.7	// the probability that a migration happens between two islands
#define migrationNumber 4			// the number of chromosomes that migrate between two islands - is supposed to be less than a half of subPopSize to avoid concurrency hazards

// CROSSOVER CONTROL PARAMETERS
#define alpha 0.2	// environmental advantage
#define beta 0.7	// base hospitality
// The probability that a crossover happens on a certain island is in the range [alpha/(popSize/subPopSize) + beta , alpha + beta] - a minimum of beta is granted for each island

// MUTATION CONTROL PARAMETERS
#define gamma 0.2	// environmental replication disadvantage
#define delta 0.3	// base replication disadvantage
// The probability that a mutation happens on a certain island is in the range [gamma/(popSize/subPopSize) + delta , gamma + delta] - a minimum of delta is granted for each island

#define RandomnessPatience 1000	// number of iterations before a deterministic gene is selected in the crossover

// A generic debug last error output for cuda
void checkCUDAError(const char *msg = NULL) {
	cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
		printf("Error after call to %s\n", msg);
    }
}

// Device function to shuffle a chromosome
__device__ void device_random_shuffle(int *chromosome, int thread){
	curandState state;
	curand_init((unsigned long long)clock64()+threadIdx.x,0,0,&state);	// init random seed
	int gene;
	for(int i=0;i<pathSize;i++){
		int idx1 = int(int(curand_uniform(&state)*pathSize) % pathSize);
		if (idx1 == pathSize){
			idx1 = pathSize-1;
		}
		int idx2 = int(int(curand_uniform(&state)*pathSize) % pathSize);	// random from 0 to 1 * path size
		if (idx2 == pathSize){
			idx2 = pathSize-1;
		}
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

// Calculate the total distance of a chromosome / TSP solution
__device__ double device_calculate_distance(int *chromosome, double *distance_matrix){
	double distance = 0.0;
	for (int i = 0; i < pathSize-1; i++){
		distance += distance_matrix[chromosome[i]*pathSize + chromosome[i+1]];
	}

	distance += distance_matrix[chromosome[pathSize-1]*pathSize + chromosome[0]];
	return distance;
}

// A kernel to calculate the distance and fitness of each chromosome in the population
__global__ void calculate_scores(int *population, double *distance_matrix,  double *population_fitness, double *population_distances){
	int island_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize * pathSize;
	int sub_pop_fitness_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize;
	// avoid calculation if island exceeds the number of islands
	if (island_index < popSize*pathSize){
		// calculate the distance and fitness for each individual in the subpopulation
		for (int i = 0; i < subPopSize; i++){
			population_distances[sub_pop_fitness_index+i] = device_calculate_distance(population+island_index+i*pathSize, distance_matrix);
			population_fitness[sub_pop_fitness_index+i] = 10000 / population_distances[sub_pop_fitness_index+i];
		}
	}
}

// A kernel to sort the subpopulations based on the fitness
__device__ void device_fit_sort_subpop(int* sub_population, double* sub_population_fitness){
	// other algorithms could be employed to narrow the granularity but bubblesort should do it's job on a small enough subpopulation
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
			}
		}
	}
}

// Crossover of two chromosomes
__device__ void device_distance_crossover(int *parent1, int *parent2, int *offspring, double *distance_matrix, unsigned long long seed){
	curandState state;
	curand_init(seed,0,0,&state);	// init random numbers generator
	// select a random point in parent 1
	int first = int(int(curand_uniform(&state)*pathSize) % pathSize);
	if (first == pathSize){
		first = pathSize-1;
	}
	// use it as first part of the offspring
	offspring[0] = parent1[first];
	// starting from the first point in both parents, compare the distance with the previous point in the offspring and choose the one with the smallest distance if it is not already in the offspring
	// if both are already in the offspring, choose a random one not in the offspring from parent 1
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
			int next = parent1[int(curand_uniform(&state)*pathSize) % pathSize];
			bool in_offspring = true;
			int iter = 0;
			while (in_offspring){
				in_offspring = false;
				for (int j = 0; j < offspring_size; j++){
					if (offspring[j] == next){
						next = parent1[int(curand_uniform(&state)*pathSize) % pathSize];
						in_offspring = true;
						break;
					}
				}
				iter++;
				if(iter > RandomnessPatience){
					// Randomly picking a gene is taking too long, iterate over parent 1 and chose one that is not in the offspring
					for (int g = 0; g < pathSize; g++){
						next = parent1[g];
						in_offspring = false;
						for (int j = 0; j < offspring_size; j++){
							if (offspring[j] == next){
								in_offspring = true;
								break;
							}
						}
						if (!in_offspring){
							break;
						}
					}
				}
			}
			offspring[offspring_size] = next;
			offspring_size++;
			i++;
			continue;
		}else if (in_offspring1 && !in_offspring2){
			offspring[offspring_size] = next2;
			offspring_size++;
		}else if (in_offspring2 && !in_offspring1){
			offspring[offspring_size] = next1;
			offspring_size++;
		}else if(!in_offspring1 && !in_offspring2){
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

// Crossover of fittest chromosomes in the subpopulation - Novel procedure "Fittest Roulette"
__device__ void device_crossover(int *sub_population, double *distance_matrix){
	int island = blockIdx.x * threadsPerBlock + threadIdx.x;
	// for this subpopulation define the crossover rate
	double crossover_rate = (alpha * (island+1))/(popSize/subPopSize) + beta;
	
	// init the random seed
	curandState state;
	unsigned long long seed = (unsigned long long)clock64()+threadIdx.x;
	curand_init(seed,0,0,&state);	// init random seed

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
		// if all the non-mating chromosomes are already overwritten, decrease the remaining top percent to overwrite the last mating chromosomes
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
			device_distance_crossover(sub_population+i*pathSize, sub_population+(i+1)*pathSize, sub_population+j*pathSize, distance_matrix, seed);
			new_borns += 1;
			j -= 1;
		}
		i += 2;	// move to the next pair of chromosomes
	}
}

// A kernel to perform the Selection and Crossover steps of the GA
__global__ void genetic_step(int *population, double *population_fitness, double *distance_matrix){
	int fitness_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	int sub_pop_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
	if (fitness_index < popSize){
		// Selection - Sort the subpopulation
		device_fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
		// Crossover
		device_crossover(population+sub_pop_index, distance_matrix);
	}
}

// A kernel to perform the mutation step of the GA
__global__ void mutation(int *population){
	int island = blockIdx.x * threadsPerBlock + threadIdx.x;
	// avoid calculation if island exceeds the number of islands
	if (island < popSize/subPopSize){
		double mutation_rate = (gamma * (island+1))/(popSize/subPopSize) + delta;
		// mutate with a certain probability each individual in the subpopulation except the first one
		curandState state;
		curand_init((unsigned long long)clock64()+island,0,0,&state);	// init random seed
		for(int i = 1; i < subPopSize; i++){
			if (curand_uniform(&state) < mutation_rate){
				// mutate the chromosome swapping two random adjacent genes
				int idx1 = int(curand_uniform(&state)*pathSize) % pathSize;// random from 0 to path_size -1
				int idx2 = (idx1 + 1) % pathSize;
				int temp = population[island*subPopSize*pathSize+i*pathSize+idx1];
				population[island*subPopSize*pathSize+i*pathSize+idx1] = population[island*subPopSize*pathSize+i*pathSize+idx2];
				population[island*subPopSize*pathSize+i*pathSize+idx2] = temp;
			}
		}
	}
}

// A kernel to perform the migration step of the Island modelled GA
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
	// NOTE: We ASSUME that migrationNumber is always less than half of the subpopulation size to avoid write before read conflicts
}

// Sort the subpopulations based on the fitness
__global__ void fit_sort(int *population, double *population_fitness){
	int fitness_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	int sub_pop_index = (blockIdx.x * threadsPerBlock + threadIdx.x) * subPopSize * pathSize;	// index of the first chromosome of each subpopulation

	if (fitness_index >= popSize){
		return;
	}
	// Sort the subpopulation
	device_fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
}

// Helper function to check if the individuals in the population are unique
__global__ void check_uniqueness_of_individuals(int *population){
	// for each chromosome in the population check if it contains unique genes
	// if not, print the chromosome
	int island = blockIdx.x * threadsPerBlock + threadIdx.x;
	if (island >= popSize/subPopSize){
		return;
	}
	for (int i = 0; i < subPopSize; i++){
		bool unique = true;
		for (int j = 0; j < pathSize; j++){
			for (int k = j+1; k < pathSize; k++){
				if (population[island*subPopSize*pathSize+i*pathSize+j] == population[island*subPopSize*pathSize+i*pathSize+k]){
					unique = false;
					break;
				}
			}
			if (!unique){
				printf("Chromosome %d in island %d is not unique and contains gene '%d'  twice\n", i, island, population[island*subPopSize*pathSize+i*pathSize+j]);
				// print the chromosome
				for (int g = 0; g < pathSize; g++){
					printf("%d ", population[island*subPopSize*pathSize+i*pathSize+g]);
				}
				printf("\n");
				break;
			}
		}
	}
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

// Helper function to save the best solution to a file
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



int main(){
	clock_t start = clock();

	// Load the coordinates of the cities from file
    int *path_coordinates = (int*)malloc(pathSize * 2 * sizeof(int));
    load_data(path_coordinates, "../data/48_cities.txt");

	// Allocate and fill the distance matrix in RAM
	double *distance_matrix = (double*)malloc(pathSize*pathSize*sizeof(double));	//[pathSize][pathSize];
	for(int i = 0; i < pathSize; i++){
		for(int j = 0; j < pathSize; j++){
			distance_matrix[i*pathSize+j] = sqrt(pow(path_coordinates[i*2]-path_coordinates[j*2],2) + pow(path_coordinates[i*2+1]-path_coordinates[j*2+1],2));
		}
	}

	// Allocate and fill the population in RAM - Matrix of indices of the cities
	int *population = (int*)malloc(popSize * pathSize * sizeof(int));
	for (int i = 0; i < pathSize * popSize; i++){
		population[i] = i % pathSize;
		
	}

	// Instantiate the gpu_population in the VRAM
	int *gpu_population;
	cudaMalloc(&gpu_population, popSize * pathSize * sizeof(int));
	cudaMemcpy(gpu_population, population, popSize * pathSize * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate fitness and distances for each individual in RAM
	double *population_fitness = (double*)malloc(popSize*sizeof(double));	//[popSize];
	double *population_distances = (double*)malloc(popSize*sizeof(double));	//[popSize];
	// these are needed in RAM for the final selection but are used in the GPU for the rest of the operations

	// Allocate fitness and distances for each individual in VRAM
	double *gpu_distance_matrix;
	cudaMalloc(&gpu_distance_matrix, pathSize*pathSize*sizeof(double));
	cudaMemcpy(gpu_distance_matrix, distance_matrix, pathSize*pathSize*sizeof(double), cudaMemcpyHostToDevice);
	double *gpu_population_fitness;		// for now just an empty array
	cudaMalloc(&gpu_population_fitness, popSize*sizeof(double));
	double *gpu_population_distances;	// for now just an empty array
	cudaMalloc(&gpu_population_distances, popSize*sizeof(double));

	// Two different granularity levels for grids and block - Only the random shuffle is done at chromosome granularity
	dim3 c_grid(popSize/subPopSize,1,1);	dim3 c_block(subPopSize,1,1);									// chromosome granularity
	dim3 s_grid(ceil((popSize/subPopSize) * (1.0/threadsPerBlock)),1,1); dim3 s_block(threadsPerBlock,1,1); // subpopulation granularity

	printf("Using %d total threads in subpopulation granularity\n", s_grid.x*s_block.x);
	printf("Of which %d blocks containing %d threads each\n", s_grid.x, s_block.x);

	// Take a checkpoint for the loading of the data to measure overhead of bus transfer
	clock_t load_checkpoint = clock();
	fprintf(stderr,"Data loaded in %.2f ms\n", ((double) (load_checkpoint - start)) * 1000.0 / CLOCKS_PER_SEC);

	// Random shuffle the population in the GPU for the first time
	random_shuffle<<<c_grid,c_block>>>(gpu_population);					// chromosome granularity
    // Check for any errors launching the kernel
	checkCUDAError("random_shuffle kernel launch");

    cudaDeviceSynchronize();

	// execute the fitness + distance kernel	(fitness can be defined as the inverse of the distance or equivalently as alpha/distance where alpha is a constant)
	calculate_scores<<<s_grid,s_block>>>(gpu_population, gpu_distance_matrix, gpu_population_fitness, gpu_population_distances);

	checkCUDAError("calculate_scores kernel launch");

	cudaDeviceSynchronize();

	//------------------------------------------------------------------------------------------------------------------------------
	// Start the actual GA -- Note: GAs are intrinsically sequential: selection -> crossover -> mutation -> (migration) -> next gen
	// in practice the parallelization takes place within single steps
	//------------------------------------------------------------------------------------------------------------------------------
	int generation = 1;
	srand(time(NULL));	// seed the random number generator
	printf("Starting the GA\n");
	while(generation <= generations){
		clock_t start_gen = clock();

		// Execute Selection and Crossover
		genetic_step<<<s_grid,s_block>>>(gpu_population, gpu_population_fitness, gpu_distance_matrix);
			checkCUDAError("genetic_step kernel launch");
		cudaDeviceSynchronize();

		// Mutate new generation
		// NOTE: this step is now outside the genetic_step kernel due to previous experiments with lower granularity
		mutation<<<s_grid,s_block>>>(gpu_population);
			checkCUDAError("mutation kernel launch");
		cudaDeviceSynchronize();
		
		// If it's time, attempt migration
		if (generation % migrationAttemptDelay == 0 && rand()/RAND_MAX < migrationProbability){
			// Migration - we suppose when a single island is able to migrate it forces the other islands to migrate as well
			migration<<<s_grid,s_block>>>(gpu_population);
				checkCUDAError("migration kernel launch");
			cudaDeviceSynchronize();
		}

		// Realign the population fitness and distances for the new generation
		calculate_scores<<<s_grid,s_block>>>(gpu_population, gpu_distance_matrix, gpu_population_fitness, gpu_population_distances);
			checkCUDAError("calculate_scores kernel launch");
		cudaDeviceSynchronize();

		clock_t end_gen = clock();
		fprintf(stderr,"Generation %d completed in %.2f ms\n", generation, ((double) (end_gen - start_gen)) * 1000.0 / CLOCKS_PER_SEC);
		generation++;
	}
	// Sort the final population one more time
	fit_sort<<<s_grid,s_block>>>(gpu_population, gpu_population_fitness); // subpopulation granularity
	checkCUDAError("fit_sort kernel launch");
	// Make sure the individuals in the population are unique - this is a debugging step
	// check_uniqueness_of_individuals<<<s_grid,s_block>>>(gpu_population);	// subpopulation granularity
	cudaDeviceSynchronize();

	// Move back to host memory the final population
	cudaMemcpy(population, gpu_population, popSize * pathSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(population_fitness, gpu_population_fitness, popSize*sizeof(double), cudaMemcpyDeviceToHost);
	// Considering each island (subpopulation) are sorted, select the best amog all the islands
	double best_fitness = 0.0;
	int best_index = 0;
	for (int i = 0; i < popSize; i += subPopSize){	// O(popSize/subPopSize) = O(number of islands) complexity
		if (population_fitness[i] > best_fitness){
			best_fitness = population_fitness[i];
			best_index = i;
		}
	}

	// Extract the best chromosome from the population
	int *best_chromosome = (int*)malloc(pathSize*sizeof(int));
	for (int i = 0; i < pathSize; i++){
		best_chromosome[i] = population[best_index*pathSize+i];
	}

	save_best_solution(best_chromosome, path_coordinates);

	// Calculate the total distance of the best chromosome
	double best_distance = 0.0;
	for (int i = 0; i < pathSize-1; i++){
		best_distance += distance_matrix[best_chromosome[i]*pathSize + best_chromosome[i+1]];
	}
	best_distance += distance_matrix[best_chromosome[pathSize-1]*pathSize + best_chromosome[0]];

	// Print the best chromosome and the distance
	printf("Best chromosome: ");
	for (int i = 0; i < pathSize; i++){
		printf("%d ", best_chromosome[i]);
	}
	printf("\nBest distance: %f\n", best_distance);


	// Free the memory
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
	printf("\nExecution completed in %.2f ms\n", ((double) (end - start)) * 1000.0 / CLOCKS_PER_SEC);

	// exit
	cudaDeviceReset();
	return 0;
}