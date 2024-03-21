#include<stdio.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<stdio.h>
#include<time.h>

clock_t start;
clock_t end;



const int popsize = 100;
const int city = 31;

const int BlockDim = 1;
const int ThreadDim = 5;
const int Eachblock = 20;
const int SN = Eachblock*0.5;
const int EX = Eachblock*0.5;	// percentage of the subpopulation that migrates
const double ER = 0.5;			// probability of migration
const double PM = 0.5;			// probability of mutations
const int GAALL = 100;
const int GAGEN = 100;

__device__ int T = 1;



__device__ void execute_random_shuffle(int a[]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	int temp;
	for(int i=0;i<city;i++){
		int rand1 = int(curand_uniform(&state)*city);
		int rand2 = int(curand_uniform(&state)*city);
		temp = a[rand1];
		a[rand1] = a[rand2];
		a[rand2] = temp;
	}
}

__global__ void random_shuffle(int pop_gpu[][city]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	int index = thread*Eachblock;
	for(int i=0;i<Eachblock;i++){
	execute_random_shuffle(pop_gpu[index+i]);
	}
}

void city_distance(int city_info[][2],double dis[][city],int num){
	for(int i=0;i<num-1;i++){
		for(int j=i+1;j<num;j++){
			int x1 = city_info[i][0];
			int x2 = city_info[j][0];
			int y1 = city_info[i][1];
			int y2 = city_info[j][1];
			dis[i][j] = pow((y2-y1),2)+pow((x2-x1),2);
			dis[i][j] = sqrt(dis[i][j]);
			dis[j][i] = dis[i][j];
		}
	}
}

__device__ double execute_cal_length(int pop_gpu[],double city_dis_gpu[][city]){
		float temp = 0;
		for(int i=0;i<city-1;i++){
			temp = temp + city_dis_gpu[pop_gpu[i]][pop_gpu[i+1]];
		}
		temp = temp + city_dis_gpu[pop_gpu[0]][pop_gpu[city-1]];
		return temp;
}

__global__ void cal_length_fit(int pop_gpu[][city],double city_dis_gpu[][city],double pop_dis_gpu[],double pop_fit_gpu[]){
		int thread = blockIdx.x * ThreadDim + threadIdx.x;
		int index = thread*Eachblock;
		for(int i=0;i<Eachblock;i++){
		pop_dis_gpu[index+i] = execute_cal_length(pop_gpu[index+i],city_dis_gpu);
		pop_fit_gpu[index+i] = 100/pop_dis_gpu[index+i];
		}
}

__device__ void sortbyfit(int pop[][city],double pop_fit_gpu[],int index){
	float temp;
	int temp2[city];
	for(int i=index;i<index+Eachblock-1;i++){
		for(int j=i+1;j<index+Eachblock;j++){
			if(pop_fit_gpu[i]>pop_fit_gpu[j]){
				temp = pop_fit_gpu[i];
				pop_fit_gpu[i] = pop_fit_gpu[j];
				pop_fit_gpu[j] = temp;
				for(int k=0;k<city;k++){
					 temp2[k] = pop[i][k];
					 pop[i][k] = pop[j][k];
					 pop[j][k] = temp2[k];
				}

			}
		}
	}
}

__device__ void selection(int pop_gpu[][city],double pop_fit_gpu[],int index){
	int j = index+Eachblock-1;
	for(int i=index;i<index+SN;i++){
		pop_fit_gpu[i] = pop_fit_gpu[j];
		for(int k=0;k<city;k++){
			pop_gpu[i][k] = pop_gpu[j][k];
		}
		j--;
	}
}

__device__ void change(int a[],int b[],int num){
	int temp[city];
		int j=0;
		for(int i=0;i<city;i++){
			int flag = 0;
			for(int k=0;k<num;k++){
				if(a[i]==b[k]){
					flag = 1;
				}
			}
			if(flag == 0){
				temp[j] = a[i];
				j++;
			}
			}
		int m=0;
		for(int i=j;i<city;i++){
			temp[i] = b[m];
			m++;
		}
		for(int i=0;i<city;i++){
			a[i] = temp[i];
		}
}

__device__ void execute_crossover(int pop1[],int pop2[]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	int position = int(curand_uniform(&state)*city);
	//int position = 2;
	int num = city-position;
	int seg1[city];
	int seg2[city];
	int k =0;
	for(int i=position;i<city;i++){
		seg1[k] = pop1[i];
		seg2[k] = pop2[i];
		k++;
	}
	change(pop1,seg2,num);
	change(pop2,seg1,num);

}

__device__ void crossover(int pop_gpu[][city],double pop_fit_gpu[],int index){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	int j=0;
	for(int i=index;i<(index+SN)/2;i++){
		float cross_rand = curand_uniform(&state);
		if(cross_rand<1){
			//printf("%d \n",index);
			execute_crossover(pop_gpu[i],pop_gpu[index+SN-1-j]);
		}
		j--;
	}
}

__device__ void ro(int a[],int len,int num){
	int pos;
	for(int i=0;i<len;i++){
		if(a[i]==num){
			pos = i;
		}
	}
	int temp[city];
	int j = 0;
	for(int i=pos;i<len;i++){
		temp[j] = a[i];
		j++;
	}
	for(int k=0;k<pos;k++){
		temp[j] = a[k];
		j++;
	}
	for(int i=0;i<len;i++){
		a[i] = temp[i];
	}
}

__device__ void hga(int a[],int b[],double dis[][city]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	int res[city];
	int num = int(curand_uniform(&state)*city);
	ro(a,city,num);
	ro(b,city,num);
	res[0] = a[0];
	for(int i=0;i<city;i++){
		if(dis[(a+i)[0]][(a+i)[1]]<dis[(b+i)[0]][(b+i)[1]]){
			res[i+1] = (a+i)[1];
			ro(b+i+1,city-i-1,(a+i)[1]);
		}else{
			res[i+1] = b[i+1];
			ro(a+i+1,city-i-1,(b+i)[1]);
		}
	}
	for(int i=0;i<city;i++){
		a[i] = res[i];
	}
	int cha = city/2;
	int j=0;
	for(int i=cha;i<city;i++){
		b[j] = res[i];
		j++;
	}
	for(int k=0;k<cha;k++){
		b[j] = res[k];
		j++;
	}

}

__device__ void new_crossover(int pop_gpu[][city],int index,double city_dis_gpu[][city]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	int j = 0;
	for(int i=index;i<index+SN/2;i++){
		float cross_rate = curand_uniform(&state);
		if(cross_rate<0.8){
		hga(pop_gpu[i],pop_gpu[index+SN-1-j],city_dis_gpu);
		}
		j--;
	}
}


__device__ void mutation(int pop_gpu[][city],int index){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	double PMU = 2*PM*T/10*GAALL;
		for(int i=index;i<index+SN;i++){
				double muta_rate = curand_uniform(&state);
				if(muta_rate<PMU){
				//printf("%d %f\n",i,muta_rate);
				int pos1 = int(curand_uniform(&state)*city);
				int pos2 = int(curand_uniform(&state)*city);
				int temp;
				if(pos1!=pos2){
					temp = pop_gpu[i][pos1];
					pop_gpu[i][pos1] = pop_gpu[i][pos2];
					pop_gpu[i][pos2] = temp;
				}
				}
		}
}

__device__ void execute_exc(int pop_gpu[][city],double pop_fit_gpu[],int index1,int index2){
	int j = index2+Eachblock-1;
	for(int i=index1;i<index1+EX;i++){
		pop_fit_gpu[j] = pop_fit_gpu[i];
		for(int k=0;k<city;k++){
			pop_gpu[j][k] = pop_gpu[i][k];
		}
		j--;
	}
}

__global__ void exchange(int pop_gpu[][city],double pop_fit_gpu[]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	int index = thread*Eachblock;
	curandState state;
	curand_init((unsigned long long)clock()+thread,0,0,&state);
	double exc_rate = curand_uniform(&state);
	if(exc_rate<ER){
		execute_exc(pop_gpu,pop_fit_gpu,index,(index+Eachblock)%popsize);	// cyclic migration: from N-1 to N, from N to 0, from 0 to 1 ...
	}
}



__device__ void findfit(double pop_fit_gpu[]){
	double max = pop_fit_gpu[0];
	int index;
	for(int i=0;i<popsize;i++){
		if(pop_fit_gpu[i]>max){
			max = pop_fit_gpu[i];
			index = i;
		}
	}
	printf("the best fit: %f\n",pop_fit_gpu[index]);
}

__global__ void GA(int pop_gpu[][city],double pop_fit_gpu[],double city_dis_gpu[][city]){
	int thread = blockIdx.x * ThreadDim + threadIdx.x;
	int index = thread*Eachblock;
	//printf("%d \n",T);
	sortbyfit(pop_gpu,pop_fit_gpu,index);
	selection(pop_gpu,pop_fit_gpu,index);
	//crossover(pop_gpu,pop_fit_gpu,index);
	new_crossover(pop_gpu,index,city_dis_gpu);
	mutation(pop_gpu,index);
//	}

}

__global__ void best(int pop_gpu[][city],double pop_dis_gpu[]){
	double min = pop_dis_gpu[0];
	for(int i=0;i<popsize;i++){
		if(pop_dis_gpu[i]<min){
			min = pop_dis_gpu[i];
		}
	}
	printf("the best length: %f \n",min);
}


__global__ void test(double a[][city]){
	for(int i=0;i<city;i++){
		for(int j=0;j<city;j++){
		printf("%f ",a[i][j]);
	}
	printf("\n");
}
}

__global__ void cal_gen(){
	T = T+1;
}

void checkCUDAError(const char *msg = NULL) {
	cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
		printf("Error after call to %s\n", msg);
        // Additional error handling if needed
    }
}


int main(){
	start = clock();
	int pop[popsize][city];
//	int city_co[city][2] = {{1,1},{1,5},{4,1},{4,5}};
//	int city_co[city][2] = {{0,0},{12,32},{5,25},{8,45},
//				{33,17},{25,7},{15,15},{15,25},{25,15},{41,12}
//				};
		int city_co[city][2]={{1304,2312},{3639,1315},{4177,2244},{3712,1399},{3488,1535},
		{3326,1556},{3238,1229},{4196,1004},{4312,790},{4386,570},{3007,1970},{2562,1756},
		{2788,1491},{2381,1676},{1332,695},{3715,1678},{3918,2179},{4061,2370},{3780,2212},{3676,2578},
		{4029,2838},{4263,2931},{3429,1908},{3507,2367},{3394,2643},{3439,3201},{2935,3240},{3140,3550},
		{2545,2357},{2778,2826},{2370,2975}};

	double city_dis[city][city];
	double pop_dis[popsize];
	double pop_fit[popsize];


//GPU::create city sequence
	for(int i=0;i<popsize;i++){
		for(int j=0;j<city;j++){
			pop[i][j] = j;
		}
	}
	int (*pop_gpu)[city];
	cudaMalloc((void**)&pop_gpu,popsize*city*sizeof(int));
	cudaMemcpy(pop_gpu,pop,popsize*city*sizeof(int),cudaMemcpyHostToDevice);
	random_shuffle<<<BlockDim,ThreadDim>>>(pop_gpu);

//CPU::calculate the distance between each city
	city_distance(city_co,city_dis,city);


//GPU::calculate the distance and fitness in every population
	double (*city_dis_gpu)[city];
	double *pop_dis_gpu;
	double *pop_fit_gpu;
	cudaMalloc((void**)&city_dis_gpu,city*city*sizeof(double));
	cudaMalloc((void**)&pop_dis_gpu,popsize*sizeof(double));
	cudaMalloc((void**)&pop_fit_gpu,popsize*sizeof(double));
	cudaMemcpy(city_dis_gpu,city_dis,city*city*sizeof(double),cudaMemcpyHostToDevice);
	cal_length_fit<<<BlockDim,ThreadDim>>>(pop_gpu,city_dis_gpu,pop_dis_gpu,pop_fit_gpu);



	cudaMemcpy(pop_fit,pop_fit_gpu,popsize*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(pop_dis,pop_dis_gpu,popsize*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(pop,pop_gpu,city*popsize*sizeof(int),cudaMemcpyDeviceToHost);

	for(int i=0;i<popsize;i++){
		printf("%f  %f ",pop_fit[i],pop_dis[i]);
		for(int j=0;j<city;j++){
			printf("%d ",pop[i][j]);
		}
		printf("\n");
	}
	for(int i=0;i<GAGEN;i++){
		for(int i=0;i<GAALL;i++){
			GA<<<BlockDim,ThreadDim>>>(pop_gpu,pop_fit_gpu,city_dis_gpu);
			cal_length_fit<<<BlockDim,ThreadDim>>>(pop_gpu,city_dis_gpu,pop_dis_gpu,pop_fit_gpu);
			cal_gen<<<1,1>>>();
		}
		exchange<<<BlockDim,ThreadDim>>>(pop_gpu,pop_fit_gpu);
	}

	cal_length_fit<<<BlockDim,ThreadDim>>>(pop_gpu,city_dis_gpu,pop_dis_gpu,pop_fit_gpu);
	best<<<1,1>>>(pop_gpu,pop_dis_gpu);
	cudaMemcpy(pop_fit,pop_fit_gpu,popsize*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(pop_dis,pop_dis_gpu,popsize*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(pop,pop_gpu,city*popsize*sizeof(int),cudaMemcpyDeviceToHost);

	printf("after ga\n");
	for(int i=0;i<popsize;i++){
		printf("%f  %f ",pop_fit[i],pop_dis[i]);
		for(int j=0;j<city;j++){
			printf("%d ",pop[i][j]);
	}
		printf("\n");
	}

	end = clock();
	printf("time : %f\n",(double)(end-start)/CLOCKS_PER_SEC);

}

