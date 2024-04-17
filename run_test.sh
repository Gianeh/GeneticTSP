#!/bin/bash
files=("./src/Island_GA_single_core.c" "./src/Island_GA_multi_core.cpp" "./src/Island_GA_multi_core_no_pool.cpp" "./src/Island_GA_opemp.cpp" "./src/Island_GA_cuda.cu")
pop_sizes=(128 1024 8192 65536 524288)
generations=100
cuda_tpbs=(32 64)

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
            g++  -o3 -o "$exec_name" "$file"
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

cd src

for exec in "${executables[@]}";do
    for((i=1;i<6;i++)); do
        new_exec=$(echo "$exec" | cut -d"/" -f3)
        echo "executing $new_exec for the $i time"
        result=$("./$new_exec")
        echo "$result"
    done
done