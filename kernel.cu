
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <chrono> 



__global__ void CalcDD(float* realAnglesArray, int* DD, unsigned long long int *counter, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {

		int j;
		float pi = atanf(1) * 4;
		float angleGalaxies;
		float alpha1;
		float alpha2;
		float delta1;
		float delta2;
		float x;
		int arrayIndex;


		//D
		alpha1 = realAnglesArray[2 * i] * pi / (60 * 180);
		delta1 = realAnglesArray[2 * i + 1] * pi / (60 * 180);

		float sinfDelta1 = sinf(delta1);
		float cosfDelta1 = cosf(delta1);

		//Histogram DD:
		for (j = 0; j < N; j++) {
			//D
			alpha2 = realAnglesArray[2 * j] * pi / (60 * 180);
			delta2 = realAnglesArray[2 * j + 1] * pi / (60 * 180);

			//angle between two galaxies a1d1 a2d2 in degrees
			// x is used to make sure that precision errors never cause acosf() to return NaN if x>1
			x = sinfDelta1 * sinf(delta2) + cosfDelta1 * cosf(delta2) * cosf(alpha1 - alpha2);
			if (x > 1) {
				x = 1;
			}
			if (x < -1) {
				x = -1;
			}
			angleGalaxies = acosf(x) * 180 / pi;
			arrayIndex = 4 * angleGalaxies;
			//adds to counter for every addition to histogram
			atomicAdd(counter, 1);
			atomicAdd(&DD[arrayIndex], 1);
		}
	}
}
__global__ void CalcRR(float* syntheticAnglesArray, int* RR, unsigned long long int *counter, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {

		int j;
		float pi = atanf(1) * 4;
		float angleGalaxies;
		float alpha1;
		float alpha2;
		float delta1;
		float delta2;
		float x;
		int arrayIndex;


		//R
		alpha1 = syntheticAnglesArray[2 * i] * pi / (60 * 180);
		delta1 = syntheticAnglesArray[2 * i + 1] * pi / (60 * 180);

		float sinfDelta1 = sinf(delta1);
		float cosfDelta1 = cosf(delta1);

		//Histogram RR:
		for (j = 0; j < N; j++) {
			//R
			alpha2 = syntheticAnglesArray[2 * j] * pi / (60 * 180);
			delta2 = syntheticAnglesArray[2 * j + 1] * pi / (60 * 180);

			//angle between two galaxies a1d1 a2d2 in degrees
			// x is used to make sure that precision errors never cause acosf() to return NaN if x>1
			x = sinfDelta1 * sinf(delta2) + cosfDelta1 * cosf(delta2) * cosf(alpha1 - alpha2);
			if (x > 1) {
				x = 1;
			}
			if (x < -1) {
				x = -1;
			}
			angleGalaxies = acosf(x) * 180 / pi;
			arrayIndex = 4 * angleGalaxies;
			//adds to counter for every addition to histogram
			atomicAdd(counter, 1);
			atomicAdd(&RR[arrayIndex], 1);
		}
	}
}

__global__ void CalcDR(float* realAnglesArray, float* syntheticAnglesArray, int* DR, unsigned long long int *counter, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N) {

		int j;
		float pi = atanf(1) * 4;
		float angleGalaxies;
		float alpha1;
		float alpha2;
		float delta1;
		float delta2;
		float x;
		int arrayIndex;


		//D
		alpha1 = realAnglesArray[2 * i] * pi / (60 * 180);
		delta1 = realAnglesArray[2 * i + 1] * pi / (60 * 180);

		float sinfDelta1 = sinf(delta1);
		float cosfDelta1 = cosf(delta1);

		//Histogram DR:
		for (j = 0; j < N; j++) {
			//R
			alpha2 = syntheticAnglesArray[2 * j] * pi / (60 * 180);
			delta2 = syntheticAnglesArray[2 * j + 1] * pi / (60 * 180);

			//angle between two galaxies a1d1 a2d2 in degrees
			// x is used to make sure that precision errors never cause acosf() to return NaN if x>1
			x = sinfDelta1 * sinf(delta2) + cosfDelta1 * cosf(delta2) * cosf(alpha1 - alpha2);
			if (x > 1) {
				x = 1;
			}
			if (x < -1) {
				x = -1;
			}
			angleGalaxies = acosf(x) * 180 / pi;
			arrayIndex = 4 * angleGalaxies;
			//adds to counter for every addition to histogram
			atomicAdd(counter, 1);
			atomicAdd(&DR[arrayIndex], 1);
		}
	}
}


int main()

{	
	//timer to measure performance
	using namespace std::chrono;

	auto start = high_resolution_clock::now();

	//number of threads for histogram calculation (N)
	const int N = 100000;
	//each galaxy has 2 angles
	const int angleAmount = 2*N;
	int i;
	float pi = atanf(1) * 4;
	int histogramIndex;


	//Variables for calculating statistics
	float DDf;
	float DRf;
	float RRf;
	float omega[360];

	float *realAnglesArray;
	float *syntheticAnglesArray;

	//histograms 90 DEG, 0.25 spacing
	int *DD;
	int *RR;
	int *DR;


	//counter to keep track of the number of histogram additions (10 000 000 000 x 3 if N = 100 000)
	unsigned long long int *counter;


	size_t arraybytes = angleAmount * sizeof(float);
	size_t harraybytes = 360 * sizeof(int);
	size_t counterbytes = sizeof(unsigned long long int);

	// Allocate using unified pointers
	cudaMallocManaged((void **)&realAnglesArray, arraybytes);
	cudaMallocManaged((void **)&syntheticAnglesArray, arraybytes);
	cudaMallocManaged((void **)&DD, harraybytes);
	cudaMallocManaged((void **)&RR, harraybytes);
	cudaMallocManaged((void **)&DR, harraybytes);
	cudaMallocManaged((void **)&counter, counterbytes);
	*counter = 0;

	// initialize arrays DD, RR, DR
	for (int i = 0; i < 360; i++) {
		DD[i] = 0;
		RR[i] = 0;
		DR[i] = 0;
	}

	printf("Reading data from files:\n\n");
	FILE *angles;
	angles = fopen("flat_100k_arcmin.txt", "r");

	//read synthetic data file into array. 200 000 length
	if (angles == NULL) {
		printf("Error Reading File\n");
		exit(0);
	}

	for (i = 0; i < angleAmount; i++) {
		fscanf(angles, "%f", &syntheticAnglesArray[i]);
	}

	fclose(angles);

	angles = fopen("data_100k_arcmin.txt", "r");

	//read real data file into array. 200 000 length
	if (angles == NULL) {
		printf("Error Reading File\n");
		exit(0);
	}

	for (i = 0; i < angleAmount; i++) {
		fscanf(angles, "%f", &realAnglesArray[i]);
	}

	fclose(angles);	


	// Invoke kernel
	int threadsInBlock = 256;
	int blocksInGrid = (N + threadsInBlock - 1) / threadsInBlock;
	
	printf("\nCalculating DD...");
	CalcDD << <blocksInGrid, threadsInBlock >> > (realAnglesArray, DD, counter, N);
	cudaDeviceSynchronize();
	printf("\nDone. Counter (number of additions): %llu", *counter);

	printf("\nCalculating RR...");
	CalcRR << <blocksInGrid, threadsInBlock >> > (syntheticAnglesArray, RR, counter, N);
	cudaDeviceSynchronize();
	printf("\nDone. Counter (number of additions): %llu", *counter);

	printf("\nCalculating DR...");
	CalcDR << <blocksInGrid, threadsInBlock >> > (realAnglesArray, syntheticAnglesArray, DR, counter, N);
	cudaDeviceSynchronize();
	printf("\nDone. Counter (number of additions): %llu", *counter);

	//print results
	for (i = 0; i < 360; i++) {
		histogramIndex = i / 4;
		printf("\nHistogram DD: i: %d  Value: %d", histogramIndex, DD[i]);
	}
	for (i = 0; i < 360; i++) {
		histogramIndex = i / 4;
		printf("\nHistogram RR: i: %d  Value: %d", histogramIndex, RR[i]);
	}
	for (i = 0; i < 360; i++) {
		histogramIndex = i / 4;
		printf("\nHistogram DR: i: %d  Value: %d", histogramIndex, DR[i]);
	}

	//calculating statistics
	for (i = 0; i < 360; i++) {
		DDf = DD[i];
		DRf = DR[i];
		RRf = RR[i];
		omega[i] = (DDf - 2 * DRf + RRf) / RRf;
		printf("\nOmega: i: %d  Value: %f", i, omega[i]);
	}


	//stop timer 
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	printf("\nRun time (ms): %d", duration.count()/1000);

	return 0;
}

/*
19.2.2020:
threadsInBlock: 512
Run time (ms): 24610, 21200, 21615
threadsInBlock: 256
Run time (ms): 21668, 21138

threadsInBlock: 256, no atomicadd counter:
Run time (ms): 17093, 16956, 17171

Last values of DR: 
Histogram DR: i: 88  Value: 715978
Histogram DR: i: 89  Value: 528683
Histogram DR: i: 89  Value: 363505
Histogram DR: i: 89  Value: 214580
Histogram DR: i: 89  Value: 69851
*/