#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define generations 50		// number of generations

#define pathSize 48				// dataset size in number of coordinates
#define popSize 1024			// population size
#define subPopSize 32				// popSize must be a multiple of this and this should be a multiple of the warp size (32)
#define selectionThreshold 0.5		// the threshold (%) for the selection of the best chromosomes in the sub-populations
#define migrationAttemptDelay 10	// the number of generations before a migration attempt is made
#define migrationProbability 0.7	// the probability that a migration happens between two islands
#define migrationNumber 4			// the number of chromosomes that migrate between two islands - is supposed to be less than a half of subPopSize to avoid concurrency issues

// CROSSOVER CONTROL PARAMETERS
#define alpha 0.2	// environmental advantage
#define beta 0.7	// base hospitality
// The probability that a crossover happens on a certain island is in the range [alpha/(popSize/subPopSize) + beta , alpha + beta] - a minimum of beta is granted for each island

// MUTATION CONTROL PARAMETERS
#define gamma 0.2	// environmental replication disadvantage
#define delta 0.3	// base replication disadvantage
// The probability that a mutation happens on a certain island is in the range [gamma/(popSize/subPopSize) + delta , gamma + delta] - a minimum of delta is granted for each island

#define RandomnessPatience 1000  // number of iterations before a deterministic gene is selected in the crossover

// Shuffle the entire population
void random_shuffle(int *population){
    for (int i = 0; i < popSize; i++){
        // shuffle the ith chromosome
        int gene;
        for (int j = 0; j < pathSize; j++){
            int idx1 = (int)((rand()/(RAND_MAX+1.0)) * pathSize);
            int idx2 = (int)((rand()/(RAND_MAX+1.0)) * pathSize);
            gene = population[i*pathSize + idx1];
            population[i*pathSize + idx1] = population[i*pathSize + idx2];
            population[i*pathSize + idx2] = gene;
        }
    }
}

// Calculate total distance of a chromosome
double calculate_distance(int *chromosome, double *distance_matrix){
    double distance = 0.0;
	for (int i = 0; i < pathSize-1; i++){
		distance += distance_matrix[chromosome[i]*pathSize + chromosome[i+1]];
	}
	distance += distance_matrix[chromosome[pathSize-1]*pathSize + chromosome[0]];
	return distance;
}

// Calculate the fitness of each chromosome in the population
void calculate_scores(int *population, double *distance_matrix, double *population_fitness, double *population_distances){
    for (int i = 0; i < popSize; i++){
        population_distances[i] = calculate_distance(population + i * pathSize, distance_matrix);
        population_fitness[i] = 1000/population_distances[i];
    }
}

// Sort the subpopulation according to the fitness of the chromosomes
void fit_sort_subpop(int *sub_population, double *sub_population_fitness){
    for (int i = 0; i < subPopSize; i++){
		for (int j = 0; j < subPopSize-1; j++){
			if (sub_population_fitness[j] < sub_population_fitness[j+1]){
				// swap the fitness
				double temp = sub_population_fitness[j];
				sub_population_fitness[j] = sub_population_fitness[j+1];
				sub_population_fitness[j+1] = temp;

				// swap all the genes in the chromosome sequentially
				for (int k = 0; k < pathSize; k++){
					int temp = sub_population[j*pathSize+k];
					sub_population[j*pathSize+k] = sub_population[(j+1)*pathSize+k];
					sub_population[(j+1)*pathSize+k] = temp;
				}
			}
		}
	}
}

// Crossover two chromosomes
void distance_crossover(int *parent1, int *parent2, int *offspring, double *distance_matrix){
    // select a random point in parent 1
	int first = (int)((rand()/(RAND_MAX+1.0)) * pathSize);
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
		int in_offspring1 = 0;	// boolean
		int in_offspring2 = 0;
		for (int j = 0; j < offspring_size; j++){
			if (offspring[j] == next1){
				in_offspring1 = 1;
			}
			if (offspring[j] == next2){
				in_offspring2 = 1;
			}
			if (in_offspring1 && in_offspring2){
				break;
			}
		}
		if (in_offspring1 && in_offspring2){
			// choose a random one not in the offspring
			int next = parent1[(int)((rand()/(RAND_MAX+1.0)) * pathSize)];
			int in_offspring = 1;
			int iter = 0;
			while (in_offspring){
				in_offspring = 0;
				for (int j = 0; j < offspring_size; j++){
					if (offspring[j] == next){
						next = parent1[(int)((rand()/(RAND_MAX+1.0)) * pathSize)];
						in_offspring = 1;
						break;
					}
				}

				iter++;
				if(iter > RandomnessPatience){
					// randomly picking a gene is taking too long, iterate over parent 1 and chose one that is not in the offspring
					for (int g = 0; g < pathSize; g++){	// parent1 is already passed with offset i+1 considering the first city is already in the offspring
						next = parent1[g];
						in_offspring = 0;
						for (int j = 0; j < offspring_size; j++){
							if (offspring[j] == next){
								in_offspring = 1;
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

// Crossover fittest chromosomes in all the subpopulations - novel method "Fittest Roulette"
void crossover(int *sub_population, double *distance_matrix, int island_index){
    // for this subpopulation define the crossover rate
	double crossover_rate = (alpha * (island_index+1))/(popSize/subPopSize) + beta;

    // starting from the top down, each pair of chromosomes i,i+1 is crossed and the offspring takes the place of the last chromosome in the subpopulation in ascending order
	// Fittest Roulette!
    int new_borns = 0;
	int i = 0;
	int j = subPopSize-1;
	int remaining_top_percent = (int)(subPopSize*selectionThreshold);
	// if odd number of selected chromosomes, increase the remaining top percent to let mate the best of the excluded
	if ((int)(subPopSize*selectionThreshold) % 2 == 1){
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

        if (rand()/(RAND_MAX + 1.0) < crossover_rate){
			// crossover the two chromosomes
			distance_crossover(sub_population+i*pathSize, sub_population+(i+1)*pathSize, sub_population+j*pathSize, distance_matrix);
			new_borns += 1;
			j -= 1;
		}
		i += 2;	// move to the next pair of chromosomes
	}
    
}

//nSelection and Crossover
void genetic_step(int *population, double *population_fitness, double *distance_matrix){
    for (int i = 0; i < popSize/subPopSize; i++){
        int fitness_index = i * subPopSize;	// index of the first fitness of each subpopulation - is the same for the distances
	    int sub_pop_index = i * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
        // Selection - Sort subpopulations according to fittness
        fit_sort_subpop(population+sub_pop_index, population_fitness+fitness_index);
	    // Crossover
        crossover(population+sub_pop_index, distance_matrix, i);
    }
}

// Mutation of new born chromosomes
void mutation(int *population){
	for (int i = 0; i < popSize/subPopSize; i++){
		int sub_pop_index = i * subPopSize * pathSize;	// index of the first chromosome of each subpopulation
		double mutation_rate = (gamma * (i+1))/(popSize/subPopSize) + delta;
		// for each chromosome in the subpopulation except the first one, apply mutation with the mutation rate
		for (int j = 1; j < subPopSize; j++){
			if (rand()/(RAND_MAX + 1.0) < mutation_rate){
				// select two random points in the chromosome and swap them
				int idx1 = (int)((rand()/(RAND_MAX+1.0)) * pathSize);
				int idx2 = (idx1+1) % pathSize;
				int temp = population[sub_pop_index+j*pathSize+idx1];
				population[sub_pop_index+j*pathSize+idx1] = population[sub_pop_index+j*pathSize+idx2];
				population[sub_pop_index+j*pathSize+idx2] = temp;
			}
		}
	}
}

// Migration between subpopulations
void migration(int *population){
	for (int i = 0; i < popSize/subPopSize; i++){
		int sub_pop_index = i;
		int next_sub_pop_index = (sub_pop_index+1) % (popSize/subPopSize);
		// swap the chromosome i from subpopulation sub_pop_index with the chromosome subPopSize-migrationNumber+i from subpopulation next_sub_pop_index
		for (int j = 0; j < migrationNumber; j++){
			// swap the chromosome i from subpopulation sub_pop_index with the chromosome subPopSize-migrationNumber+i from subpopulation next_sub_pop_index
			for (int k = 0; k < pathSize; k++){
				int temp = population[sub_pop_index*subPopSize*pathSize+j*pathSize+k];
				population[sub_pop_index*subPopSize*pathSize+j*pathSize+k] = population[next_sub_pop_index*subPopSize*pathSize+(subPopSize-migrationNumber+j)*pathSize+k];
				population[next_sub_pop_index*subPopSize*pathSize+(subPopSize-migrationNumber+j)*pathSize+k] = temp;
			}
		}
	}
}

// Sort the subpopulations according to the fitness of the chromosomes
void fit_sort(int *population, double *population_fitness){
	for (int i = 0; i < popSize/subPopSize; i++){
		fit_sort_subpop(population+i*subPopSize*pathSize, population_fitness+i*subPopSize);
	}
}

void load_data(int *coordinates, const char *filename){
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

int main(){
    clock_t start = clock();

	// Load the coordinates of the cities from file
    int *path_coordinates = (int*)malloc(pathSize * 2 * sizeof(int));	//[pathSize][2];
    load_data(path_coordinates, "../data/48_cities.txt");

	// Allocate and fill the distance matrix
	double *distance_matrix = (double*)malloc(pathSize*pathSize*sizeof(double));	//[pathSize][pathSize];
	for(int i = 0; i < pathSize; i++){
		for(int j = 0; j < pathSize; j++){
			distance_matrix[i*pathSize+j] = sqrt(pow(path_coordinates[i*2]-path_coordinates[j*2],2) + pow(path_coordinates[i*2+1]-path_coordinates[j*2+1],2));
		}
	}

	// Allocate and fill the population
    int *population = (int*)malloc(popSize * pathSize * sizeof(int));	//[popSize][pathSize]; - This represents the order of the cities for each chromosome in the population
	for (int i = 0; i < pathSize * popSize; i++){
		population[i] = i % pathSize;
		
	}

	// Allocate fitness and distances for each individual and calculate them
	double *population_fitness = (double*)malloc(popSize*sizeof(double));	//[popSize];
	double *population_distances = (double*)malloc(popSize*sizeof(double));	//[popSize];

    clock_t load_checkpoint = clock();
	printf("Data loaded in %.2f ms\n", ((double) (load_checkpoint - start)) * 1000.0 / CLOCKS_PER_SEC);


	// Random shuffle the population for the first time and calculate the fitness and distances
    srand(time(NULL));	// seed the random number generator

    random_shuffle(population);

    calculate_scores(population, distance_matrix, population_fitness, population_distances);


    //------------------------------------------------------------------------------------------------------------------------------
	// Start the actual GA -- Note: GAs are intrinsically sequential: selection -> crossover -> mutation -> (migration) -> next gen
	//------------------------------------------------------------------------------------------------------------------------------
	int generation = 1;
    while (generation <= generations){
        clock_t start_gen = clock();

		// Selection and Crossover
        genetic_step(population, population_fitness, distance_matrix);

		// Mutation
		mutation(population);

		// If it's time, attempt migration
		if (generation % migrationAttemptDelay == 0 && rand()/(RAND_MAX + 1.0) < migrationProbability){
			migration(population);
		}

		// Calculate the fitness and distances of the new population
		calculate_scores(population, distance_matrix, population_fitness, population_distances);

		clock_t end_gen = clock();
		printf("Generation %d completed in %.2f ms\n", generation, ((double) (end_gen - start_gen)) * 1000.0 / CLOCKS_PER_SEC);
		generation++;
    }

	// Sort the population according to the fitness of the chromosomes one last time
	fit_sort(population, population_fitness);
	
	// Considering each island (subpopulation) are sorted, select the best amog all the islands
	double best_fitness = 0.0;
	int best_index = 0;
	for (int i = 0; i < popSize; i += subPopSize){
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

	// Calculate the distance of the best chromosome
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

	// Free the memory
	free(path_coordinates);
	free(distance_matrix);
	free(population);
	free(population_fitness);
	free(population_distances);
	free(best_chromosome);

	clock_t end = clock();
	fprintf(stderr,"Execution completed in %.2f ms\n", ((double) (end - start)) * 1000.0 / CLOCKS_PER_SEC);

	// exit
	return 0;
}