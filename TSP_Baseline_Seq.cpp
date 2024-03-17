// This file implements the baseline sequential solution for the TSP problem.
// This is hence parallelized in TSP_Baseline_par.cpp using the chosen parallelization strategy from paper:
// "A comparative study of five parallel genetic algorithms using the traveling salesman problem"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// Solution class, basically the single genome/path of the TSP problem
// using the Krolak datasets require modeling cities as pairs of doubles - future datasets may require distance matrices or graphs
class TSPSolution {
    public:
        TSPSolution(const std::vector<std::pair<double, double>>& cities) : path(cities){};
        void shufflePath() {
            std::random_shuffle(path.begin() + 1, path.end());
        }
        
        double calculateFitness() const {
            return 1.0 / calculateTotalDistance();
        }

        std::vector<std::pair<double, double>> getPath() const {
            return path;
        }

        std::pair<double, double> getRandomCity() const {
            return path[rand() % path.size()];
        }

        void printSolution() const {
            std::cout << "Path: ";
            for (auto& city : path) {
                std::cout << "(" << city.first << ", " << city.second << ") ";
            }
            std::cout << std::endl;
            // print the total distance of the path
            std::cout << "Total distance: " << calculateTotalDistance() << std::endl;
        }
        double calculateTotalDistance() const {
            double totalDistance = 0.0;
            for (int i = 0; i < path.size() - 1; ++i) {
                totalDistance += calculateDistance(path[i], path[i + 1]);
            }
            totalDistance += calculateDistance(path.back(), path.front());
            return totalDistance;
        }

    private:
        //std::vector<std::pair<double, double>> cities;
        std::vector<std::pair<double, double>> path;
        double calculateDistance(const std::pair<double, double>& city1, const std::pair<double, double>& city2) const {
            double dx = city1.first - city2.first;
            double dy = city1.second - city2.second;
            return std::sqrt(dx * dx + dy * dy);
        }
        

        
};

// Class Population handles the genetic algorithm operations
// Following the cited peper, the algorithm applies 3 different operations to a given percentage of the population at each iteration (generation)
// x% of the population is selected for crossover (sequentially adding subpaths from the best parent to the child)
// y% of the population is mutated through 2-opt technique (randomly swapping 2 subpaths depending on the distance between the cities)
// z% of the population is mutated through or-opt technique

class Population {
    public:
        Population(int population_size, const std::vector<std::pair<double, double>>& cities, const float crossover_rate=100.0, const float opt_2_rate=0.0, const float opt_or_rate=0.0) :
        populationSize(population_size), crossoverRate(crossover_rate), opt2Rate(opt_2_rate), optOrRate(opt_or_rate) {
            // Initialize the population with random paths
            for (int i = 0; i < populationSize; ++i) {
                TSPSolution path(cities);
                path.shufflePath();
                genomes.push_back(path);
            }
        }

        // The function that actually applies the changes to the current population of genomes to create the next generation
        void evolve() {
            // newGenomes is cleared to store the new generation
            newGenomes.clear();

            // Population is shuffled to avoid picking always the same genomes for crossover or mutation
            std::random_shuffle(genomes.begin(), genomes.end());

            // crossover_rate% of the population is selected for crossover
            int crossoverCount = (int)(populationSize * crossoverRate / 100);
            //std::cout << "Crossover count: " << crossoverCount << std::endl;
            crossover(crossoverCount);

            // opt_2_rate% of the population is selected for 2-opt mutation
            int opt2Count = (int)(populationSize * opt2Rate / 100);
            //std::cout << "Opt2 count: " << opt2Count << std::endl;
            opt2(crossoverCount + 1, opt2Count);

            // opt_or_rate% of the population is selected for or-opt mutation
            int optOrCount = (int)(populationSize * optOrRate / 100);
            //std::cout << "OptOr count: " << optOrCount << std::endl;
            optOr(crossoverCount + opt2Count + 1, optOrCount);

            return;
        }

        void crossover(int crossoverCount) {
            // Pair the parents for crossover in both the cases of even and odd crossoverCount
            std::vector<std::pair<TSPSolution,TSPSolution>> families;
            for (int i = 0; i < crossoverCount/2; i++) {
                // Considering the population is already shuffled we add sequential pairs of genomes to the parents vector
                families.push_back(std::make_pair(genomes[i], genomes[crossoverCount/2+i]));
            }
            // if odd
            if (crossoverCount % 2 != 0) {
                families.push_back(std::make_pair(genomes[crossoverCount-1], genomes[rand() % crossoverCount]));
            }

            // log the count of families
            std::cout << "Families count: " << families.size() << std::endl;
            
            // Crossover Heuristic operation
            for (auto& parents : families){
                // Create 1 child from the parents
                std::vector<std::pair<double, double>> path;
                // Select a random starting point from the first parent
                path.push_back(parents.first.getRandomCity());
                // Sequentially compare:
                /*the Heuristic
                crossover operation extends the current partially
                constructed offspring tour by trying to add the shorter
                parental next city from the current last city (c) - on this
                partial offspring tour. When the next city of c on both
                parent tours is already on the current partial offspring
                tour, a random city is selected from those that are not yet
                on the partial offspring tour*/
                for (int i = 0; i < parents.first.getPath().size()-1; i++){
                    double d1 = 0.1;    // just a random value to avoid standard 0.0
                    double d2 = 0.1;
                    // Check if the city from first parent is already in the child
                    if (std::find(path.begin(), path.end(), parents.first.getPath()[i]) != path.end()){
                        d1 = 0.0;
                    }
                    // Check if the city from second parent is already in the child
                    if (std::find(path.begin(), path.end(), parents.second.getPath()[i]) != path.end()){
                        d2 = 0.0;
                    }
                    // Compare the distance between the last city added to the child and the next city in both parents
                    d1 = d1 == 0.0 ? 0.0 : calculateDistance(path.back(), parents.first.getPath()[i]);
                    d2 = d2 == 0.0 ? 0.0 : calculateDistance(path.back(), parents.second.getPath()[i]);

                    // Add the city with the shortest distance to the child
                    if (d1 == 0.0 && d2 == 0.0){
                        // select a random city from those that are not yet on the partial offspring tour
                        while (true){
                            std::pair<double, double> city = parents.first.getRandomCity();
                            if (std::find(path.begin(), path.end(), city) == path.end()){
                                path.push_back(city);
                                break;
                            }
                        }
                    }
                    else if (d2 == 0.0 && d1 != 0.0){
                        path.push_back(parents.first.getPath()[i]);
                    }
                    else if (d1 == 0.0 && d2 != 0.0){
                        path.push_back(parents.second.getPath()[i]);
                    }
                    else if (d1 <= d2){
                        path.push_back(parents.first.getPath()[i]);
                    }
                    else if (d2 < d1){
                        path.push_back(parents.second.getPath()[i]);
                    }
                }
                // Keep the best parent based on the fitness function
                TSPSolution bestParent = parents.first.calculateFitness() >= parents.second.calculateFitness() ? parents.first : parents.second;
                // Make parent decision explicit
                /*
                std::cout << "Best parent: " << std::endl;
                bestParent.printSolution();
                // among the two
                std::cout << "Parents: " << std::endl;
                parents.first.printSolution();
                parents.second.printSolution();
                */
                TSPSolution child(path);

                // Add to the new generation
                newGenomes.push_back(child);
                newGenomes.push_back(bestParent);
            }

            // Add the new generation to the population
            for (int i = 0; i < crossoverCount; i++){
                genomes[i] = newGenomes[i];
            }
        }

        // 2-opt mutation
        // This function implements the 2-opt mutation heuristic, according to the paper:
        /*
        The 2-opt step [3] randomly selects one half of the
        remaining chromosomes of the population (25% of the
        total chromosomes). On each selected tour, 10 2-opt
        attempts are made to try to shorten the tour length, each
        time randomly picking two pairs of adjacent cities on the
        tour. The 2-opt operator takes these two pairs of
        adjacent cities (a, b) and (c, d) from a tour. If
        dab + dcd > d, + dbd, then the paths between cities a, b
        and cities c, d on the tour are removed and replaced by
        the new paths between cities a, c and cities b, d.
        */
        void opt2(int startIndex, int opt2Count){
            for (auto& genome : genomes) {
                // for ten times pick two pairs of adjacent cities on the tour
                for (int i = 0; i < 10; i++){
                    // generate two random indices
                    int a = rand() % genome.getPath().size()-1; // -1 to avoid the last city
                    int b = a+1;
                    int c = rand() % genome.getPath().size()-1;
                    int d = c+1;
                    // make sure c != a, a+1, a-1
                    while (c == a || c == a+1 || c == a-1){
                        c = rand() % genome.getPath().size()-1;
                    }
                    std::pair<double,double> cityA = genome.getPath()[a];
                    std::pair<double,double> cityB = genome.getPath()[b];
                    std::pair<double,double> cityC = genome.getPath()[c];
                    std::pair<double,double> cityD = genome.getPath()[d];
                    // calculate the distances
                    double ab = calculateDistance(cityA, cityB);
                    double cd = calculateDistance(cityC, cityD);
                    double ac = calculateDistance(cityA, cityC);
                    double bd = calculateDistance(cityB, cityD);  
                    // if dab + dcd > dac, + dbd, then the paths between cities a, b and cities c, d on the tour are removed and replaced by the new paths between cities a, c and cities b, d.
                    if (ab + cd >= ac + bd){
                        //get the min and max indeces into wich we have to flip the cities
                        std::vector<int> path = {a, b, c, d};
                        minind=path[0];
                        maxind=path[0];
                        for(i=0;i<path.size();i++){
                            if(path[i] < minind){
                                minind=path[i];
                            }
                            else if(path[i] > maxind){
                                maxind=path[i];
                            }
                        }
                        //flip the cities
                        genome.flip(minind,maxind);
                    }

                }
            }
        }

        void optOr(int startIndex, int optOrCount){
            return;
        }

        void printBestSolution() {
            // Sort the genomes based on their fitness and print the best solution
            TSPSolution bestSolution = genomes[0];
            for (auto& genome : genomes) {
                if (genome.calculateFitness() > bestSolution.calculateFitness()) {
                    bestSolution = genome;
                }
            }
            bestSolution.printSolution();
        }

        void printInfo() {
            std::cout << "Population size: " << populationSize << std::endl;
            std::cout << "Crossover rate: " << crossoverRate << std::endl;
            std::cout << "Opt2 rate: " << opt2Rate << std::endl;
            std::cout << "OptOr rate: " << optOrRate << std::endl;
        }
        
        void print() {
            for (auto& genome : genomes) {
                std::cout << genome.calculateTotalDistance() << std::endl;
            }
        }


    private:
        int populationSize;
        float crossoverRate;
        float opt2Rate;
        float optOrRate;
        std::vector<TSPSolution> genomes;
        std::vector<TSPSolution> newGenomes;

        double calculateDistance(const std::pair<double, double>& city1, const std::pair<double, double>& city2) const {
            double dx = city1.first - city2.first;
            double dy = city1.second - city2.second;
            return std::sqrt(dx * dx + dy * dy);
        }

        
};


int main(){
    srand(42);
    // 11 cities:    
    std::vector<std::pair<double, double>> cities = {{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0}, {4.0, 3.0}, {2.0, 4.0}, {5.0, 2.0}, {6.0, 3.0}, {7.0, 1.0}, {8.0, 4.0}, {9.0, 0.0}, {10.0, 2.0}};
    // optimal distance: 25.879248604912885
    Population population(100, cities, 100.0, 0.0, 0.0);
    // Print algorithm information and initiate the genetic search
    std::cout << "Initiating Genetic search:" << std::endl;
    //population.printInfo();
    for (int i = 0; i < 100; ++i) {
        population.evolve();
        std::cout << "\t\t\t\tGeneration: " << i+1 << std::endl;
        //std::cout << "all the genomes in the population:" << std::endl;
        //population.print();
        population.printBestSolution();
    }
    return 0;

}