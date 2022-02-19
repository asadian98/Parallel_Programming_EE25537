# Parallel merge sort algorithm using pthread

Parallelized merge sort algorithm using pthread library. I used 4 thread of executions to perform 4 ‘merge’ jobs in parallel (one thread for each of the 4 sort modules).  Each sort module is itself a merge sort. It works for any array size N=2^M (24 <= M <= 30). I used serial merge algorithm for the 2 middle merge modules. Besides, I launched one thread for each of the 2 middle merge modules (total of 2 threads for all middle merge). The last ‘merge’ was parallelized using 4 threads in this step.

![image](https://user-images.githubusercontent.com/94138466/154815102-6edd5a05-3aca-40e2-bbab-0abebe3afa6f.png)

``` cpp
Compile: gcc -O2 pth_msort_test.c pth_msort.c -lpthread -lm
Execute: ./a.out M
```
