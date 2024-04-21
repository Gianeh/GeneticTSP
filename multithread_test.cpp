#include <thread>
#include <chrono>
#include <iostream>
#include <vector>

#define threads 10
#define len 10000000
#define runs 10000

void task(int* vect, int portion){
    for(int i = 0; i < portion; i++){
        vect[i] = i;
        }
}

int main(){

    // long int vector
    int *a = (int*)malloc(len * sizeof(int));
    int portion = len / threads;

    // vector of threads
    std::vector<std::thread> thread_vector(threads);

    // measure elapsed time
    auto start = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < runs; r++){
        for(int i = 0; i < threads; i++){
            thread_vector[i] = std::thread(task, a + i * portion, portion);
        }
        // join threads
        for(int i = 0; i < threads; i++){
            thread_vector[i].join();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end - start;
    // elapsed time in milliseconds
    std::cout << "Elapsed time: " << elapsed_time.count() * 1000 << " ms" << std::endl;

    // DO EVERYTHING AGAIN IN SINGLE THREAD

    // measure elapsed time
    start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; r++)
        task(a, len);
    end = std::chrono::high_resolution_clock::now();
    elapsed_time = end - start;
    // elapsed time in milliseconds
    std::cout << "Elapsed time: " << elapsed_time.count() * 1000 << " ms" << std::endl;

    free(a);

    return 0;
}