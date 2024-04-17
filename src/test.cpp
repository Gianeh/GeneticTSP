#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <unistd.h> // Per usleep su Unix

class ThreadPool {
public:
    ThreadPool(int numThreads) : stop(false), completedTasks(0), totalTasks(0) {
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back([this, i] {
                //std::cout << "Thread " << i << " started" << std::endl;
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                    {
                        std::lock_guard<std::mutex> lock(completedTasksMutex);
                        ++completedTasks;
                        if (completedTasks == totalTasks) {
                            condition.notify_all();
                        }
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& thread : workers) {
            thread.join();
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(completedTasksMutex);
        condition.wait(lock, [this] { return tasks.empty() && completedTasks == totalTasks; });
        completedTasks = 0; // Reset completed tasks after they have all finished
    }

    void setTotalTasks(int numTasks) {
        totalTasks = numTasks;
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::mutex completedTasksMutex;
    std::condition_variable condition;
    bool stop;
    int completedTasks;
    int totalTasks;
};

void task1() {
    std::cout << "A";
}

void task2() {
    std::cout << "B";
}

int main() {
    ThreadPool pool(10);
    pool.setTotalTasks(10);
    for (int j = 0; j < 10; j++) {
        pool.enqueue([] { task1(); });
    }
    pool.wait();

    pool.setTotalTasks(10);
    for (int j = 0; j < 10; j++) {
        pool.enqueue([] { task2(); });
    }
    pool.wait();

    return 0;
}
