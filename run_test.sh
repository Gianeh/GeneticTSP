#!/bin/bash
files=( "./src/Island_GA_openmp.cpp" "./src/Island_GA_cuda.cu")
pop_sizes=(128 1024 8192 65536 524288)
generations=100
cuda_tpbs=(32 64)
result_file=$(pwd)/results/last_result.txt
rm "$result_file" #this assures that only the last result file is stored
function replace(){
    file=$1
    text_to_search=$2
    number_to_replace=$3
    sed -i -E "s/^#define $text_to_search [0-9]+/#define $text_to_search $number_to_replace/" "$file"
    parameter_changed=$(echo "$text_to_search"| awk '{print $NF}')
    if [[ $? -eq 0 ]]; then
            echo "changed $parameter_changed in file $file to: $number_to_replace"
    else
        echo "couldn't change $parameter_changed for file: $file"
    fi

}
executables=()
for file in "${files[@]}"; do
    replace "$file" "generations" "$generations"
    for popSize in "${pop_sizes[@]}"; do
        replace "$file" "popSize" "$popSize"
        if echo "$file" | grep cuda; then
            for tpb in "${cuda_tpbs[@]}"; do
                replace "$file" "threadsPerBlock" "$tpb"
                exec_name=./src/cuda_${tpb}_${popSize}
                nvcc -o "$exec_name" "$file"
                if [[ $? -eq 0 ]]; then
                echo "executed the compilation of file $file"
                executables+=("$exec_name")
                else
                    echo "something went wrong in the compilation of file $file"
                fi
            done
        fi
        if echo "$file" | grep ".cpp"; then
            exec_name=./src/$(echo "$file" | cut -d"/" -f3 | cut -d"." -f1)_${popSize}
            if echo "$file" | grep openmp; then
                g++ -fopenmp -O3 -o "$exec_name" "$file"
            else
                g++  -O3 -o "$exec_name" "$file"
            fi

            if [[ $? -eq 0 ]]; then
                echo "executed the compilation of file $file"
                executables+=("$exec_name")
            else
                echo "something went wrong in the compilation of file $file"
            fi
        fi
        if echo "$file" | grep ".c" | grep -v pp; then
            exec_name=./src/$(echo "$file" | cut -d"/" -f3 | cut -d"." -f1)_${popSize}
            gcc -o3 -o "$exec_name" "$file" -lm
            if [[ $? -eq 0 ]]; then
                echo "executed the compilation of file $file"
                executables+=("$exec_name")
            else
                echo "something went wrong in the compilation of file $file"
            fi
        fi
    done
done
echo -e "############################\n       list of executables     \n############################"
for exec in "${executables[@]}";do
    echo "$exec"
done
echo "############################"

cd src || exit

for exec in "${executables[@]}";do
    total_generations=0
    total_intergeneration_time=0
    sum_total_time=0
    every_total=true
    best_distance=9999999
    for((i=1;i<6;i++)); do
    set -x
        #get only the filename of the exec
        new_exec=$(echo "$exec" | cut -d"/" -f3)
        #execute the code redirecting the stderr in the stdout to get every line
        echo "executing $new_exec for the $i time"
        result=$("./$new_exec" 2>&1)
    set +x
        #find the total run generations by counting the lines containing "Generation XX completed in"
        generation=$(echo "$result" | grep -ic "Generation .* completed in")
        #find the generation time by summing all the the 5 th elements (numbers) in the line "Generation XX completed in"
        gen_time=$(echo "$result" | grep -i "Generation .* completed in" | awk '{ sum += $5 } END { print sum }')
        total_intergeneration_time=$(echo "$gen_time + $total_intergeneration_time" | bc)
        total_generations=$(("$total_generations" + "$generation"))
        #find the total time in result
        total_time=$(echo "$result" | grep -i "Execution completed in" | awk '{print $4}')
        #if the total time has been foun inside the result then echo it sum it to the total between runs
        if [[ -n $total_time ]]; then
            sum_total_time=$(echo "$total_time + $sum_total_time" | bc)
        else
            #if even one time the total has not been found, we don't print the avg time because the sum is not the total
            every_total=false
        fi
        best_distance_run=$(echo "$result" | grep -i "Best distance" | awk '{print $3}')
        #if the best distance has been found in the result
        if [[ -n $best_distance_run ]]; then
            #and it is smaller than the best distances found so far
            if [[ "$(echo "$best_distance_run < $best_distance" | bc)" -eq 1 ]]; then
                #update best distance
                best_distance=$best_distance_run
            fi
        fi
    done
    echo -e "##############################################\n          results\n##############################################"
    #calculate the avg intergen time and print it together with the generations
    avg_intgen_time=$(echo "scale=2; $total_intergeneration_time / $total_generations" | bc)
    echo "$new_exec" >> "$result_file"
    echo -e "total generations: $total_generations \n total intergeneration_time $total_intergeneration_time \n avg intergen_time $avg_intgen_time \n best distance $best_distance" >> "$result_file"
    #if all the total have been found
    if $every_total; then
        #calculate the avg total time
        avg_total_time=$(echo "scale=2; $sum_total_time / 5" | bc)
        echo "avg_total_time $avg_total_time" >> "$result_file"
    fi
    cat "$result_file"
    
    echo "##############################################"
done