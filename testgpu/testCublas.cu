#include <cublas_v2.h>
#include <cstdlib>
#include <curand.h>
#include <iostream>
#include "cuda_multi_gemm_unif.cu"

//To run this code, nvcc testCublas_float.cu -lcublas -lcurand -o testCublas
// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(float *A, float *B, float *C, const int m, const int k, const int n) {
    cudaDeviceSynchronize();
    cudaError_t code3 = cudaPeekAtLastError();
    //printf("CUDA: after begin: %s\n",cudaGetErrorString(code3));

    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    //printf("parametersi %d, %d, %d, %f, %f \n", m, k, n, alf, bet);

    cudaDeviceSynchronize();
    code3 = cudaPeekAtLastError();
    //printf("CUDA: after float: %s\n",cudaGetErrorString(code3));

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();
    code3 = cudaPeekAtLastError();
    //printf("CUDA: after cublasCreate: %s\n",cudaGetErrorString(code3));


    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    cudaDeviceSynchronize();
    code3 = cudaPeekAtLastError();
    //printf("CUDA: after cublasSgemm: %s\n",cudaGetErrorString(code3));
    // Destroy the handle
    cublasDestroy(handle);
}



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
    int m = 4, k=5, n = 2, nelt = 2;
    nr_rows_A = m; 
    nr_cols_A = k;
    nr_rows_B = k; 
    nr_cols_B = n*nelt;
    nr_rows_C = m;
    nr_cols_C = n*nelt;
    nr_rows_D = n;
    nr_cols_D = n;
    nr_rows_E = m;
    nr_cols_E = n*nelt;

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

    // Fill the arrays A and B on GPU with random numbers
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            //h_A[i * nr_rows_A + j]= i * nr_rows_A + j;
            h_A[i * nr_cols_A + j]= i * nr_cols_A + j;
        }
    }
    for(int i = 0; i < nr_rows_B; ++i){
        for(int j = 0; j < nr_cols_B; ++j){
            //h_B[i * nr_rows_B + j]= 0.1*(i * nr_rows_B + j);
            h_B[i * nr_cols_B + j]= 0.1*(i * nr_cols_B + j);
        }
    }
    for(int i = 0; i < nr_rows_D; ++i){
        for(int j = 0; j < nr_cols_D; ++j){
            //h_B[i * nr_rows_B + j]= 0.1*(i * nr_rows_B + j);
            h_D[i * nr_cols_D + j]= 0.1*(i * nr_cols_D + j + 10);
        }
    }

    // Optionally we can copy the data back on CPU and print the arrays
    cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_D,h_D,nr_rows_D * nr_cols_D * sizeof(float),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t code3 = cudaPeekAtLastError();
    //printf("CUDA: after cpy: %s\n",cudaGetErrorString(code3));
    std::cout << "A =" << std::endl;
    print_matrix(h_A, nr_rows_A, nr_cols_A);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, nr_rows_B, nr_cols_B);
    std::cout << "D =" << std::endl;
    print_matrix(h_D, nr_rows_D, nr_cols_D);

    // Multiply A and B on GPU
    //gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    int lda=nr_rows_A,ldb=nr_cols_A,ldc=nr_rows_A;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;                                                             
    const float *beta = &bet;                                                              
    // Create a handle for CUBLAS                                                          
    cublasHandle_t handle;                                                                 
    cublasCreate(&handle);                                                                 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n*nelt, k, alpha, d_A, m, d_B, k, beta, d_C,m);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, d_C, m, m*n, d_D, n, 0, beta, d_E,m, m*n, nelt);
    cublasDestroy(handle);                                                                 

    std::cout << "end gpu_blas_mmul" << std::endl;

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E,d_E,nr_rows_E * nr_cols_E * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);
    std::cout << "E =" << std::endl;
    print_matrix(h_E, nr_rows_E, nr_cols_E);



    //start gemm test
    cudaStream_t stream;
    cudaStreamCreate( &stream );
    const float alpha2 = 1;
    const float beta2 = 0;
    int blockSize = 2, gridSize;
    gridSize = (int)ceil((float)m*n*nelt/blockSize);
    //cudaMemset(d_C, 0.0, m*n);
    //cuda_multi_gemm_unif(stream,'N', 'N', m, n, k, &alpha2, d_A, m, 0, d_B, k, k*n, &beta2, d_C, m, m*n, nelt, gridSize);
    cuda_multi_gemm_unif(stream,'N', 'N', m, n, n, &alpha2, d_C, m, m*n, d_D, n, 0, &beta2, d_E, m, m*n, nelt, gridSize);
    std::cout << "end gemm N N" << std::endl;
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, nr_rows_C, nr_cols_C);
    cudaMemcpy(h_E,d_E,nr_rows_E * nr_cols_E * sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "E =" << std::endl;
    print_matrix(h_E, nr_rows_E, nr_cols_E);



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

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
