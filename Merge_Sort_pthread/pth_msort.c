// Include your C header files here

#include "pth_msort.h"

unsigned int Np;
int bound1;
int bound2;
int bound3;
int* valuesArr;
int* SortedArr;

void MergeL2(unsigned int begin, unsigned int end){
	
	unsigned int Idx1 = 0;
	unsigned int Idx2 = 0;
	unsigned int offset = end - begin;
	unsigned int offset_2 = offset/2;
	
	int* A1 = (int*) malloc(offset_2*sizeof(int));
	int* A2 = (int*) malloc(offset_2*sizeof(int));
	
	memcpy(A1, &valuesArr[begin], offset_2*sizeof(int));
	memcpy(A2, &valuesArr[begin + offset_2], offset_2*sizeof(int));
	
	unsigned int i;
	
	for(i = begin; i < end; i = i + 1){
		
		if(Idx1 < offset_2 && Idx2 < offset_2){
			
			if(A1[Idx1] > A2[Idx2]){
				valuesArr[i] = A2[Idx2];
				Idx2 = Idx2 + 1;
			}
			else{
				valuesArr[i] = A1[Idx1];
				Idx1 = Idx1 + 1;
			}
		}
		else if(Idx1 == offset_2){
			valuesArr[i] = A2[Idx2];
			Idx2 = Idx2 + 1;
		}
		else
		{
			valuesArr[i] = A1[Idx1];
			Idx1 = Idx1 + 1;
		}
	}
	
	free(A1);
	free(A2);
	
}


void LastMerge(unsigned begin1, unsigned end1, unsigned begin2, unsigned end2, unsigned SortedIdx){
	
	unsigned int offset1 = end1 - begin1;
	unsigned int offset2 = end2 - begin2;
	
	unsigned int i = 0; 
	unsigned int j = 0; 
	
	while(i < offset1 && j < offset2){
		
		if(valuesArr[begin1 + i] < valuesArr[begin2 + j]){
			SortedArr[SortedIdx] = valuesArr[begin1 + i];
			i = i + 1;
		}
		else{
			SortedArr[SortedIdx] = valuesArr[begin2 + j];
			j = j + 1;
		}
		SortedIdx = SortedIdx + 1;
		
	}
	
	while(i < offset1){
		
		SortedArr[SortedIdx] = valuesArr[begin1+i];
		i = i + 1; 
		SortedIdx = SortedIdx + 1;
		
	}
	
	while(j < offset2){
		
		SortedArr[SortedIdx] = valuesArr[begin2+j];
		j = j + 1;
		SortedIdx = SortedIdx + 1;
		
	}
}

void MSortL1(int begin, int end){
	
	int j = begin;	
	int i = begin - 1;
	int endVal = valuesArr[end];
	int tmp;
	
	if(begin < end){
		for(; j <= end - 1; j++){
			if(valuesArr[j] < endVal){
				i++;
				tmp = valuesArr[i];
				valuesArr[i] = valuesArr[j];
				valuesArr[j] = tmp;
			}
		}
		tmp = valuesArr[end];
		valuesArr[end] = valuesArr[i+1];
		valuesArr[i+1] = tmp;
		
		MSortL1(begin, i);
		MSortL1(i+2, end);
	}
	
}

void* Sort1(){
	
	MSortL1(0, (Np/4)-1);

}

void* Sort2(){
	
	MSortL1(Np/4, (Np/2)-1);

}

void* Sort3(){
	
	MSortL1(Np/2, (3*Np/4)-1);

}

void* Sort4(){
	
	MSortL1(3*Np/4, Np-1);

}

void* Merge1(){
	
	MergeL2(0, Np/2);
	
}

void* Merge2(){
	
	MergeL2(Np/2, Np);
	
}

void* Last_Merge1(){
	
//	LastMerge(0, Np/4, Np/2, bound1, 0);
	if(bound2 > Np/8) 	LastMerge(0, Np/8, Np/2, bound3, 0);	
	else				LastMerge(0, bound2, Np/2, 3*Np/4, 0);
}

void* Last_Merge2(){
	
//	LastMerge(Np/4, Np/2, bound1, Np, bound1 - Np/4);
	if(bound2 > Np/4)		LastMerge(Np/8, Np/4, bound3, bound1, Np/8 + bound3 - Np/2);	
	else if(bound2 > Np/8) 	LastMerge(Np/8, bound2, bound3, 3*Np/4, Np/8 + bound3 - Np/2);
	else 					LastMerge(bound2, Np/8, 3*Np/4, bound3, bound2 + Np/4);
}

void* Last_Merge3(){
	
//	if(bound2 > Np/4)	LastMerge(bound2, Np/2, 3*Np/4, Np, bound2 + Np/4);
//	else				LastMerge(Np/4, Np/2, bound1, Np, bound1 - Np/2 + Np/4);
	if(bound2 > Np/4)		LastMerge(Np/4, bound2, bound1, 3*Np/4, Np/4 + bound1 - Np/2);	
	else if(bound2 > Np/8) 	LastMerge(bound2, Np/4, 3*Np/4, bound1, bound2 + Np/4);
	else 					LastMerge(Np/8, Np/4, bound3, bound1, Np/8 + bound3 - Np/2);
	
}

void* Last_Merge4(){
	
	//LastMerge(Np/4, Np/2, bound, Np, bound - Np/4);
	if(bound2 > Np/4)		LastMerge(bound2, Np/2, 3*Np/4, Np, bound2 + Np/4);	
	else if(bound2 > Np/8) 	LastMerge(Np/4, Np/2, bound1, Np, Np/4 + bound1 - Np/2);
	else 					LastMerge(Np/4, Np/2, bound1, Np, Np/4 + bound1 - Np/2);
	
}

unsigned int binary_search(int value, unsigned int begin, unsigned int end){

	unsigned int mid = begin + (end - begin)/2;
	if(valuesArr[mid] == value || (end - begin) == 1) 	return mid + 1;
	else if(valuesArr[mid] > value)						return binary_search(value, begin, mid);
	else												return binary_search(value, mid, end);
		
}

void mergeSortParallel(const int* values, unsigned int N, int* sorted) {
	
	pthread_t handle_sort[4];
	pthread_t handle_Merge[2];
	pthread_t handle_Last_Merge[4];
	
	Np = N;
	valuesArr = values;
	SortedArr = sorted;
	
	pthread_create(&handle_sort[0], NULL, Sort1, NULL); 
	pthread_create(&handle_sort[1], NULL, Sort2, NULL); 
	pthread_create(&handle_sort[2], NULL, Sort3, NULL); 
	pthread_create(&handle_sort[3], NULL, Sort4, NULL); 
	
	pthread_join(handle_sort[0], NULL); 
	pthread_join(handle_sort[1], NULL); 
	pthread_join(handle_sort[2], NULL); 
	pthread_join(handle_sort[3], NULL); 
	
	pthread_create(&handle_Merge[0], NULL, Merge1, NULL); 
	pthread_create(&handle_Merge[1], NULL, Merge2, NULL);

	pthread_join(handle_Merge[0], NULL); 
	pthread_join(handle_Merge[1], NULL); 
		
	/*
			    0	   N/8      N/4             N-1
		Arr1 = [        |        |                ]
		Arr2 = [                 |                ]                				

	*/
	
	bound1 = binary_search(values[Np/4], Np/2, Np);
	if(bound1 < 3*Np/4)		bound2 = binary_search(values[3*Np/4], Np/4, Np/2);
	else					bound2 = binary_search(values[3*Np/4], 0, Np/4);
	bound3 = binary_search(values[Np/8], Np/2, bound1);
		
	pthread_create(&handle_Last_Merge[0], NULL, Last_Merge1, NULL); 
	pthread_create(&handle_Last_Merge[1], NULL, Last_Merge2, NULL); 
	pthread_create(&handle_Last_Merge[2], NULL, Last_Merge3, NULL); 
	pthread_create(&handle_Last_Merge[3], NULL, Last_Merge4, NULL); 
	
	pthread_join(handle_Last_Merge[0], NULL); 
	pthread_join(handle_Last_Merge[1], NULL); 	
	pthread_join(handle_Last_Merge[2], NULL); 
	pthread_join(handle_Last_Merge[3], NULL); 
	
	//Merge_main(0, Np);
		 
	//int i;
	//for (i=0 ; i<N ; i++)
	//	sorted[i] = values[i]; 
	return;
}

