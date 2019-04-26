#include <cublas_v2.h>

void gpu_local_grad3_t(cublasHandle_t &handle, double *u, double *ur, double *us, double *ut, int nx1, double *d, double *dt, double *w, int nel);

void gpu_local_grad2_t(double *u, double *ur, double *us, int nx1, double *d, double *dt, double *w, int nel);

void gpu_local_grad3(cublasHandle_t &handle, double * ur, double *us, double *ut, double *u, int nx, double *d, double *dt, int nel);

void gpu_local_grad2(double * ur, double *us, double *u, int nx, double *d, double *dt, int nel);

void gpu_gen_int(double *jgl, double *jgt,int mp,int np,double *w);

void gpu_get_int_ptr(int *ip,int  if3d, int mx, int md, int nelt,double *jgl, double *jgt,double *wkd,int lxd,int *pjgl);

void gpu_gen_dgl(double *dgl,double *dgt,int mp,int np,double *w);

void gpu_get_dgl_ptr (int *ip,int  if3d, int mx, int md, int nelt,double *dg, double *dgt,double *wkd, int lxd, int *pdg);

void gpu_specmpn(cublasHandle_t &handle, double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int neq, int eq, bool second_eq);

void gpu_gen_dgll(double *dgl,double *dgt,int mp,int np,double *w);

void gpu_get_dgll_ptr (int ip,int  if3d, int mx, int md, int nelt,double *d, double *dt,double *wkd,int  lxd, int *pdg);

void gpu_gradl_rst(double *ur, double *us, double *ut, double *u, double *d, double *dt, int md, int nel, bool if3d,int *ip,double *wkd,int *pdg,int nx, int nxd);

void gpu_double_copy_gpu_wrapper(int glbblockSize2,double *a1,int n1,double *a2,int n2,int n);

void gpu_nekadd2(int glbblockSize2,double *a, double*b, int n);
void gpu_neksub2(int glbblockSize2,double *a, double*b, int n);

void gpu_nekcol2(int glbblockSize2,double *a, double*b, int n);




