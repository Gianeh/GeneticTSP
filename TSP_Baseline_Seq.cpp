// This file implements the baseline sequential solution for the TSP problem.
// This is hence parallelized in TSP_Baseline_par.cpp using the chosen parallelization strategy from paper:
// "A comparative study of five parallel genetic algorithms using the traveling salesman problem"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

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
        

    private:
        //std::vector<std::pair<double, double>> cities;
        std::vector<std::pair<double, double>> path;
        double calculateDistance(const std::pair<double, double>& city1, const std::pair<double, double>& city2) const {
            double dx = city1.first - city2.first;
            double dy = city1.second - city2.second;
            return std::sqrt(dx * dx + dy * dy);
        }
        double calculateTotalDistance() const {
            double totalDistance = 0.0;
            for (int i = 0; i < path.size() - 1; ++i) {
                totalDistance += calculateDistance(path[i], path[i + 1]);
            }
            totalDistance += calculateDistance(path.back(), path.front());
            return totalDistance;
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
                TSPSolution genome(cities);
                genome.shufflePath();
                genomes.push_back(genome);
            }
        }

        // The function that actually applies the changes to the current population of genomes to create the next generation
        void evolve() {
            // newGenomes is cleared to store the new generation
            newGenomes.clear();

            // Population is shuffled to avoid picking always the same genomes for crossover or mutation
            std::random_shuffle(genomes.begin(), genomes.end());

            // attempt 1:
            // save the 2 best genomes from the previous generation before applying the genetic operations (with paper parameters there would be no guarantee that the best genome would be selected for crossover and thus saved)
            TSPSolution bestSolution = genomes[0];
            TSPSolution secondBestSolution = genomes[1];
            for (auto& genome : genomes) {
                if (genome.calculateFitness() > bestSolution.calculateFitness()) {
                    secondBestSolution = bestSolution;
                    bestSolution = genome;
                }
            }
            newGenomes.push_back(bestSolution);
            newGenomes.push_back(secondBestSolution);
            // NOTE: DOING SO, THE TWO LAST EDITED (THROUGH CROSSOVER OR ONE OF THE OPT-MUTATIONS) GENOMES WON'T BE ADDED TO THE NEW GENERATION

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

            // Copy the new generation to the current generation
            //genomes = newGenomes;
            for (int i = 0; i < newGenomes.size(); i++){
                genomes[i] = newGenomes[i];
            }

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
                TSPSolution child(path);

                // Add to the new generation
                newGenomes.push_back(child);
                newGenomes.push_back(bestParent);
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

            for (int g = startIndex + 1; g < opt2Count; g++) {
                // for ten times pick two pairs of adjacent cities on the tour
                std::vector<std::pair<double,double>> path = genomes[g].getPath();
                for (int i = 0; i < 10; i++){
                    // generate two random indices
                    int a = rand() % (path.size()-2); // -2 to avoid the last city
                    int b = a+1;
                    int c = rand() % (path.size()-2);
                    // make sure c != a, a+1, a-1
                    while (c == a || c == a+1 || c == a-1){
                        c = rand() % (path.size()-2);
                    }
                    int d = c+1;

                    std::pair<double,double> cityA = path[a];
                    std::pair<double,double> cityB = path[b];
                    std::pair<double,double> cityC = path[c];
                    std::pair<double,double> cityD = path[d];
                    // calculate the distances
                    double ab = calculateDistance(cityA, cityB);
                    double cd = calculateDistance(cityC, cityD);
                    double ac = calculateDistance(cityA, cityC);
                    double bd = calculateDistance(cityB, cityD);  
                    // if dab + dcd > dac, + dbd, then the paths between cities a, b and cities c, d on the tour are removed and replaced by the new paths between cities a, c and cities b, d.
                    if (ab + cd >= ac + bd){
                        //get the min and max indeces into wich we have to flip the cities
                        /*
                        std::vector<int> subPath = {a, b, c, d};
                        int minind = subPath[0];
                        int maxind = subPath[0];
                        for(i = 0; i < subPath.size(); i++){
                            if(subPath[i] < minind){
                                minind=subPath[i];
                            }
                            else if(subPath[i] > maxind){
                                maxind=subPath[i];
                            }
                        }
                        //flip the cities
                        std::reverse(path.begin() + minind + 1, path.begin() + maxind);
                        */
                        std::pair<double,double> temp = path[b];
                        path[b] = path[c];
                        path[c] = temp;
                        
                    }
                }
                std::random_shuffle(path.begin() + 1, path.end());
                TSPSolution child(path);
                newGenomes.push_back(child);
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

        TSPSolution getBestSolution() {
            // Sort the genomes based on their fitness and print the best solution
            TSPSolution bestSolution = genomes[0];
            for (auto& genome : genomes) {
                if (genome.calculateFitness() > bestSolution.calculateFitness()) {
                    bestSolution = genome;
                }
            }
            return bestSolution;
        }
        void printInfo() {
            std::cout << "Population size: " << populationSize << std::endl;
            std::cout << "Crossover rate: " << crossoverRate << std::endl;
            std::cout << "Opt2 rate: " << opt2Rate << std::endl;
            std::cout << "OptOr rate: " << optOrRate << std::endl;
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

std::vector<std::pair<double, double>> parseCoordinates(const std::string& filename) {
    std::vector<std::pair<double, double>> coordinates;
    std::ifstream input_file(filename);
    if (!input_file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return coordinates;
    }

    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream ss(line);
        double x, y;
        if (ss >> x >> y) {
            coordinates.push_back({x, y});
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    input_file.close();
    return coordinates;
}



int main(){
    srand(42);
    // 11 cities:    
    //std::vector<std::pair<double, double>> cities = {{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0}, {4.0, 3.0}, {2.0, 4.0}, {5.0, 2.0}, {6.0, 3.0}, {7.0, 1.0}, {8.0, 4.0}, {9.0, 0.0}, {10.0, 2.0}};
    // optimal distance: 25.879248604912885
    
    // 13 cities:
    //std::vector<std::pair<double, double>> cities = {{0.5, 3.0}, {6.0, 2.0}, {7.3, 2.0}, {12.0, 4.0}, {3.0, 5.0}, {5.0, 10.0}, {5.0, 12.0}, {5.0, 0.0}, {9.0, 4.0}, {15.0, 13.0}, {10.0, 2.0}, {11.0, 3.0}, {12.0, 1.0}};
    // optimal distance: 48.56695256837755  -  Source = ?

    // Krolak A:
    std::vector<std::pair<double, double>> cities = parseCoordinates("krolak_coords.txt");
    //optimal distance: 21282

    Population population(50, cities, 20.0, 80.0, 0.0);
    // Print algorithm information and initiate the genetic search
    std::cout << "Initiating Genetic search:" << std::endl;
    //population.printInfo();

    // stop condition: 100000 generations or fitness doesn't improve for 1000 generations
    int localOptimum = 0;
    double prevBestFitness = 0.0;
    for (int i = 0; i < 100000; ++i) {
        std::cout << "\t\t\t\tGeneration: " << i+1 << std::endl;
        population.evolve();
        //std::cout << "all the genomes in the population:" << std::endl;
        //population.print();
        population.printBestSolution();
        double bestFitness = population.getBestSolution().calculateFitness();
        if (bestFitness == prevBestFitness){
            localOptimum++;
            prevBestFitness = bestFitness;
        }
        else{
            localOptimum = 0;
        }
        if (localOptimum == 1000){
            std::cerr << "Local optimum reached, stopping the search after " << i+1 << " generations." << std::endl;
            break;
        }
    }

    // print the best solution in a file
        std::ofstream output_file("best_solution.txt");
        std::vector<std::pair<double,double>> bestSolution = population.getBestSolution().getPath();
        for (auto& city : bestSolution) {
            output_file << city.first << " " << city.second << std::endl;
        }
        output_file << bestSolution[0].first << " " << bestSolution[0].second << std::endl;
        output_file.close();

    return 0;

}