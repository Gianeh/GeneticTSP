#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>

class CandidateSolution {
public:
    virtual void initialize() = 0;
    virtual double calculateFitness() = 0;
    virtual void mutate() = 0;
    virtual void crossover(const CandidateSolution& parent2, CandidateSolution& child) const = 0;
    virtual void printSolution() const = 0;
    virtual ~CandidateSolution() = default;
};

class TSPSolution : public CandidateSolution {
public:
    // Constructor with list of cities in the format of (x, y) coordinates
    TSPSolution(const std::vector<std::pair<double, double>>& cities) : cities_(cities) {}

    // Initialize the solution with a random path composed of all cities in a random order
    void initialize() override {
        path_ = cities_;
        std::random_shuffle(path_.begin() + 1, path_.end()); // Shuffle cities excluding the starting city
    }

    // Calculate the fitness of the solution, which (in this case) is the inverse of the total distance
    double calculateFitness() override {
        double totalDistance = 0.0;
        for (int i = 0; i < path_.size() - 1; ++i) {
            totalDistance += calculateDistance(path_[i], path_[i + 1]); // distance from city i to city i+1
        }
        totalDistance += calculateDistance(path_.back(), path_.front()); // Distance from last to first city (back to start)
        return 1.0 / totalDistance; // Minimize distance, maximize fitness
    }

    // Mutate the solution by shuffling the order of cities in the path (excluding the starting city)
    void mutate() override {
        std::random_shuffle(path_.begin() + 1, path_.end());
    }


    // Crossover with another solution to produce a child solution
    void crossover(const CandidateSolution& parent2, CandidateSolution& child) const override {
        const TSPSolution& tspParent2 = dynamic_cast<const TSPSolution&>(parent2);
        size_t crossoverPoint1 = rand() % (path_.size() - 1) + 1; // Exclude the starting city
        size_t crossoverPoint2 = rand() % (path_.size() - crossoverPoint1) + crossoverPoint1;

        auto& childTSP = dynamic_cast<TSPSolution&>(child); // Cast child to TSPSolution
        childTSP = *this; // Copy current solution

        for (size_t i = crossoverPoint1; i <= crossoverPoint2; ++i) {
            auto it = std::find(childTSP.path_.begin(), childTSP.path_.end(), tspParent2.path_[i]);
            std::iter_swap(childTSP.path_.begin() + i, it);
        }
    }

    void printSolution() const override {
        std::cout << "Path: ";
        for (auto& city : path_) {
            std::cout << "(" << city.first << ", " << city.second << ") ";
        }
        std::cout << std::endl;
        // print the total distance of the path
        double totalDistance = 0.0;
        for (int i = 0; i < path_.size() - 1; ++i) {
            totalDistance += calculateDistance(path_[i], path_[i + 1]);
        }
        totalDistance += calculateDistance(path_.back(), path_.front());
        std::cout << "Total distance: " << totalDistance << std::endl;
    }


    std::vector<std::pair<double, double>> get_path(){
        return cities_;
    }


private:
    std::vector<std::pair<double, double>> cities_;
    std::vector<std::pair<double, double>> path_;

    double calculateDistance(const std::pair<double, double>& city1, const std::pair<double, double>& city2) const {
        double dx = city1.first - city2.first;
        double dy = city1.second - city2.second;
        return std::sqrt(dx * dx + dy * dy);
    }
};

class Population {
public:
    Population(int populationSize, const std::vector<std::pair<double, double>>& cities)
        : populationSize_(populationSize), cities_(cities) {}

    void initializePopulation() {
        population_.clear();
        for (size_t i = 0; i < populationSize_; ++i) {
            std::unique_ptr<TSPSolution> solution = std::make_unique<TSPSolution>(cities_);
            solution->initialize();
            population_.emplace_back(std::move(solution));
        }
    }

    void evolve() {
        std::sort(population_.begin(), population_.end(),
                  [](const std::unique_ptr<CandidateSolution>& a, const std::unique_ptr<CandidateSolution>& b) {
                      return a->calculateFitness() > b->calculateFitness();
                  });

        std::vector<std::unique_ptr<CandidateSolution>> newPopulation;

        // Elitism: Keep the top solutions
        for (size_t i = 0; i < elitismSize_; ++i) {
            newPopulation.emplace_back(std::make_unique<TSPSolution>(*dynamic_cast<TSPSolution*>(population_[i].get())));
        }

        // Crossover and Mutation
        while (newPopulation.size() < populationSize_) {
            size_t parent1Index = rand() % tournamentSize_;
            size_t parent2Index = rand() % tournamentSize_;
            while (parent2Index == parent1Index) {
                parent2Index = rand() % tournamentSize_;
            }

            const CandidateSolution& parent1 = *population_[parent1Index];
            const CandidateSolution& parent2 = *population_[parent2Index];

            TSPSolution child(cities_);
            parent1.crossover(parent2, child);
            if (rand() / static_cast<double>(RAND_MAX) < mutationRate_) {
                child.mutate();
            }

            newPopulation.emplace_back(std::make_unique<TSPSolution>(child));
        }

        population_ = std::move(newPopulation);
    }

    void setElitismSize(size_t elitismSize) {
        elitismSize_ = elitismSize;
    }

    void setTournamentSize(size_t tournamentSize) {
        tournamentSize_ = tournamentSize;
    }

    void setMutationRate(double mutationRate) {
        mutationRate_ = mutationRate;
    }

    const CandidateSolution& getBestSolution() const {
        return *population_.front();
    }

private:
    size_t populationSize_;
    size_t elitismSize_ = 5;
    size_t tournamentSize_ = 20;
    double mutationRate_ = 0.5;

    std::vector<std::unique_ptr<CandidateSolution>> population_;
    std::vector<std::pair<double, double>> cities_;
};



int main() {
    // intialize random seed
    //srand(time(NULL));
    srand(42);

    // SAMPLE DATA:

    // 11 cities:
    // std::vector<std::pair<double, double>> cities = {{0.0, 0.0}, {1.0, 2.0}, {3.0, 1.0}, {4.0, 3.0}, {2.0, 4.0}, {5.0, 2.0}, {6.0, 3.0}, {7.0, 1.0}, {8.0, 4.0}, {9.0, 0.0}, {10.0, 2.0}};
    // optimal distance: 25.879248604912885

    // 13 cities:
    
    //std::vector<std::pair<double, double>> cities = {{0.5, 3.0}, {6.0, 2.0}, {7.3, 2.0}, {12.0, 4.0}, {3.0, 5.0}, {5.0, 10.0}, {5.0, 12.0}, {5.0, 0.0}, {9.0, 4.0}, {15.0, 13.0}, {10.0, 2.0}, {11.0, 3.0}, {12.0, 1.0}};
    // optimal distance: 48.56695256837755

    /*
    optimal = 291.0
    #  lau15_dist.coord.txt
    #  created by TABLE_IO.F90
    #  at 11 February 2009  11:48:31.644 AM
    #
    #  Spatial dimension M =        2
    #  Number of points N =       15
    #  EPSILON (unit roundoff) =   0.222045E-15
    #
    0.549963E-07  0.985808E-08
    -28.8733     -0.797739E-07
    -79.2916      -21.4033    
    -14.6577      -43.3896    
    -64.7473       21.8982    
    -29.0585      -43.2167    
    -72.0785      0.181581    
    -36.0366      -21.6135    
    -50.4808       7.37447    
    -50.5859      -21.5882    
    -0.135819      -28.7293    
    -65.0866      -36.0625    
    -21.4983       7.31942    
    -57.5687      -43.2506    
    -43.0700       14.5548    
    */
    std::vector<std::pair<double, double>> cities = {{0.549963E-07, 0.985808E-08}, {-28.8733, -0.797739E-07}, {-79.2916, -21.4033}, {-14.6577, -43.3896}, {-64.7473, 21.8982}, {-29.0585, -43.2167}, {-72.0785, 0.181581}, {-36.0366, -21.6135}, {-50.4808, 7.37447}, {-50.5859, -21.5882}, {-0.135819, -28.7293}, {-65.0866, -36.0625}, {-21.4983, 7.31942}, {-57.5687, -43.2506}, {-43.0700, 14.5548}};

    // Parse and load data from file
    //std::vector<std::pair<double, double>> cities;

    // Inizializzazione della popolazione
    Population tspPopulation(50000, cities);
    tspPopulation.initializePopulation();

    // Log generation info every log_rate generation
    int log_rate = 50;

    // Evolve the population
    for (int generation = 0; generation < 1000; ++generation) {
        tspPopulation.evolve();
        const CandidateSolution& bestCandidate = tspPopulation.getBestSolution();

        // Dynamically cast the best solution to TSPSolution
        const TSPSolution* bestTSPSolution = dynamic_cast<const TSPSolution*>(&bestCandidate);
        if (bestTSPSolution) {
            // Handle the best TSPSolution
            // For example, print its details
            if (!(generation % log_rate)){
                std::cout << "Generation " << generation << ": Best TSP solution found:" << std::endl;
                bestTSPSolution->printSolution();
            }
           
        } else {
            // Handle error if dynamic cast fails
            std::cerr << "Error: Failed to cast best solution to TSPSolution" << std::endl;
        }
    }

    // Visualizzazione del percorso migliore
    std::cout << "Best ";
    tspPopulation.getBestSolution().printSolution();
    // stampa l'ordine numerico dei punti rispetto all'ordine del percorso originale
    /*
    std::cout << "Path: ";

    const TSPSolution* solution = dynamic_cast<const TSPSolution*>(&tspPopulation.getBestSolution());
    for (auto& city : cities) {
        std::cout << std::find(solution->get_path().begin(), solution->get_path().end(), city) - tspPopulation.getBestSolution().path_.begin() << " ";
    }
    std::cout << std::endl;
    */

    return 0;
}
