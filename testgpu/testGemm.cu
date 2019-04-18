#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include "cuda_multi_gemm_unif.cu"
#include <ctime> //for time measurement

//To run this code, nvcc testGemm.cu -lcublas -o testGemm

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

    int m = 0;
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            std::cout << A[j * nr_rows_A + i] << " ";
            //std::cout << A[m] <<" ";
            m++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C, nr_rows_D, nr_cols_D, nr_rows_E, nr_cols_E;

    // for simplicity we are going to use square arrays
    int m = 12, k=6, n = 6, nelt = 100;
    nr_rows_A = m; 
    nr_cols_A = k;
    nr_rows_B = k; 
    nr_cols_B = n*nelt;
    nr_rows_C = m;
    nr_cols_C = n*nelt;
    nr_rows_D = n;
    nr_cols_D = m;
    nr_rows_E = m;
    nr_cols_E = m*nelt;

    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    float *h_D = (float *)malloc(nr_rows_D * nr_cols_D * sizeof(float));
    float *h_E = (float *)malloc(nr_rows_E * nr_cols_E * sizeof(float));

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C, *d_D, *d_E;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
    cudaMalloc(&d_D,nr_rows_D * nr_cols_D * sizeof(float));
    cudaMalloc(&d_E,nr_rows_E * nr_cols_E * sizeof(float));

    // Fill the arrays A, B and D on CPU 
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            h_A[i * nr_cols_A + j]= (i * nr_cols_A + j) ;
        }
    }
    for(int i = 0; i < nr_rows_B; ++i){
        for(int j = 0; j < nr_cols_B; ++j){
            h_B[i * nr_cols_B + j]= (0.01*(i * nr_cols_B + j)) ;
        }
    }
    for(int i = 0; i < nr_rows_D; ++i){
        for(int j = 0; j < nr_cols_D; ++j){
            h_D[i * nr_cols_D + j]= (0.001*(i * nr_cols_D + j + 10)) ;
        }
    }

    // Optionally we can copy the data to GPU and print the arrays
    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_D,h_D,nr_rows_D * nr_cols_D * sizeof(float),cudaMemcpyHostToDevice);
    /*std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);
    std::cout << "D =" << std::endl;
    print_matrix(h_D, nr_rows_D, nr_cols_D);*/

    // Multiply A and B on GPU using cublasSgemm

    //measure time
    clock_t begin = clock();
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;                                                 
    const float *beta = &bet;                                                              
    cublasHandle_t handle;                                                                 
    cublasCreate(&handle);                                                                 
    for(int i = 0; i<10000; i++){
    cudaMemset(d_C, 0.0, m*n*nelt);
    cudaMemset(d_E, 0.0, m*n*nelt);
    // A(m,k) * B(k,n*nelt) = C(m,n*nelt)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n*nelt, k, alpha, d_A, m, d_B, k, beta, d_C,m);
    //c(m,n)*nelt * D(m,n) = E(m,n)*nelt
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, n, alpha, d_C, m, m*n, d_D, n, 0, beta, d_E,m, m*m, nelt);
    }
    cublasDestroy(handle);     
                                                            
    clock_t end = clock();
    double elapsed_sec = double(end-begin)/CLOCKS_PER_SEC;

    std::cout << "cublasSgemm time " << elapsed_sec << ' ' << end <<' ' << begin << std::endl;
    std::cout << "cublasSgemm result" << std::endl;

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E,d_E,nr_rows_E * nr_cols_E * sizeof(float),cudaMemcpyDeviceToHost);
    /*std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);
    std::cout << "E =" << std::endl;
    print_matrix(h_E, nr_rows_E, nr_cols_E);*/



    //start gemm test

    //measure time
    clock_t begin2 = clock();
    cudaStream_t stream;
    cudaStreamCreate( &stream );
    const float alpha2 = 1;
    const float beta2 = 0;
    int blockSize = 2, gridSize;

    for(int i = 0; i< 10000; i++){
    gridSize = (int)ceil((float)m*n*nelt/blockSize);
    cudaMemset(d_C, 0.0, m*n*nelt);
    cudaMemset(d_E, 0.0, m*n*nelt);
    cuda_multi_gemm_unif(stream,'N', 'N', m, n, k, &alpha2, d_A, m, 0, d_B, k, k*n, &beta2, d_C, m, m*n, nelt, gridSize);
    gridSize = (int)ceil((float)m*m*nelt/blockSize);
    cuda_multi_gemm_unif(stream,'N', 'N', m, m, n, &alpha2, d_C, m, m*n, d_D, n, 0, &beta2, d_E, m, m*m, nelt, gridSize);
    }

    clock_t end2 = clock();
    double elapsed_sec2 = double(end2-begin2)/CLOCKS_PER_SEC;
    std::cout << "cuda_multi_gemm_unif time " <<  elapsed_sec2 << ' ' << end2 <<' ' << begin2 << std::endl;

    std::cout << "cuda_multi_gemm_unif result" << std::endl;
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    /*std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);
    cudaMemcpy(h_E,d_E,nr_rows_E * nr_cols_E * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "E =" << std::endl;
    print_matrix(h_E, nr_rows_E, nr_cols_E);*/



/*    cudaMemset(d_C, 0.0, nr_rows_A*nr_cols_B);
    cuda_multi_gemm_unif(stream,'T', 'T', nr_rows_A, nr_cols_B, nr_cols_A, &alpha2, d_A, nr_cols_A, nr_rows_A*nr_cols_A, d_B, nr_cols_B, nr_rows_B*nr_cols_B, &beta2, d_C, nr_rows_A, nr_rows_A*nr_cols_B, 1, gridSize);
    std::cout << "end gemm T T" << std::endl;
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);


    cudaMemset(d_C, 0.0, nr_rows_A*nr_cols_B);
    cuda_multi_gemm_unif(stream,'T', 'N', nr_rows_A, nr_cols_B, nr_cols_A, &alpha2, d_A, nr_cols_A, nr_rows_A*nr_cols_A, d_B, nr_rows_B, nr_rows_B*nr_cols_B, &beta2, d_C, nr_rows_A, nr_rows_A*nr_cols_B, 1, gridSize);
    std::cout << "end gemm T N" << std::endl;
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);


    cudaMemset(d_C, 0.0, nr_rows_A*nr_cols_B);
    cuda_multi_gemm_unif(stream,'N', 'T', nr_rows_A, nr_cols_B, nr_cols_A, &alpha2, d_A, nr_rows_A, nr_rows_A*nr_cols_A, d_B, nr_cols_B, nr_rows_B*nr_cols_B, &beta2, d_C, nr_rows_A, nr_rows_A*nr_cols_B, 1, gridSize);
    std::cout << "end gemm N T" << std::endl;
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);*/

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);

    return 0;
}
