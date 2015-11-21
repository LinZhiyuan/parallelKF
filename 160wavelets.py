from pycuda.autoinit import context
import pycuda.driver as cuda
import numpy as np 
from pycuda.compiler import SourceModule
import numpy
import time


#cs_length = 1024
#dim_state = 168   #Nnumber of basis
#iteration = 20
dim_obs = 1 



print cuda.mem_get_info()
start_idxs = [-42, -34, -26, -18, -10, -2, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134, 142, -42, -34, -26, -18, -10, -2, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134, 142, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134, 138, 142, 146, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148]

end_idxs = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156]

identities = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
 
basis_prototypes = [  1.22270575e-02,   3.79397218e-02,   7.14230478e-02,
         1.16239148e-01,   1.65392841e-01,   2.18649196e-01,
         2.82771856e-01,   3.56523086e-01,   3.93199279e-01,
         3.89175230e-01,   3.65774819e-01,   3.09957077e-01,
         2.50646984e-01,   1.89203747e-01,   9.95557957e-02,
        -1.26848736e-02,  -8.23923092e-02,  -1.06918983e-01,
        -1.13712941e-01,  -8.55481621e-02,  -6.74810165e-02,
        -6.19085084e-02,  -3.14902369e-02,   1.43936892e-02,
         3.84249152e-02,   4.17849336e-02,   3.60165539e-02,
         1.20736680e-02,   1.52613945e-03,   5.75903199e-03,
         1.54414750e-03,  -4.27282175e-03,  -6.86231676e-03,
        -8.28016518e-03,  -5.98115575e-03,   7.94918495e-04,
         3.35462105e-03,   1.85665743e-03,   1.20429219e-03,
        -4.12845493e-04,  -1.04692789e-03,  -1.46155765e-04,
         3.30667638e-05,   3.67410127e-05,   1.13821552e-04,
        -6.73358661e-06,  -3.24638210e-05,   7.15656321e-06,
         3.69292415e-06,  -1.19014041e-06],[ -5.62445833e-04,  -1.74523089e-03,  -3.28546713e-03,
        -5.34701211e-03,  -7.60808673e-03,  -1.00578843e-02,
        -1.30075328e-02,  -1.64000966e-02,  -1.49100952e-02,
        -8.04374433e-03,   1.73306967e-03,   1.59458211e-02,
         3.14463118e-02,   4.81109739e-02,   6.88965838e-02,
         9.32233731e-02,   9.53177278e-02,   7.30204584e-02,
         3.81092228e-02,  -1.66973803e-02,  -7.57223210e-02,
        -1.38297680e-01,  -2.18802583e-01,  -3.14270006e-01,
        -2.95668046e-01,  -1.52029073e-01,   5.78259725e-02,
         3.70458077e-01,   4.77253543e-01,   3.51289628e-01,
         1.72288227e-01,  -1.41563540e-01,  -2.76698973e-01,
        -1.98286912e-01,  -1.25964304e-01,   2.17925386e-02,
         8.69034972e-02,   3.95351590e-02,   2.21937039e-02,
        -8.09606843e-03,  -2.23057686e-02,  -3.32343990e-03,
         7.18841174e-04,   7.98715981e-04,   2.47437634e-03,
        -1.46382008e-04,  -7.05733750e-04,   1.55577133e-04,
         8.02807902e-05,  -2.58725630e-05],[-0.00244141, -0.00757552, -0.01426121, -0.02320975, -0.01165801,
        0.02264021,  0.0683477 ,  0.13193628,  0.10470012, -0.02393838,
       -0.19696322, -0.44475829, -0.22194155,  0.5258149 ,  0.51498549,
       -0.20384957, -0.29978739,  0.03517939,  0.06659498, -0.01468069,
       -0.00757552,  0.00244141],[-0.0105974 , -0.03288301,  0.03084138,  0.18703481, -0.02798377,
       -0.63088077,  0.71484657, -0.23037781]



dim_state = len(start_idxs)  #168
cs_length = 160 #max(end_idxs)    #192
#cs_length = 5
obs_mat = np.zeros((cs_length,dim_state))  #whole observation matrix

x1 = np.linspace(0, 30, 160)

t1 = np.exp(-0.02*x1)*3

for basis_idx in range(dim_state):
  for t in range(max(start_idxs[basis_idx],0),min(end_idxs[basis_idx],cs_length)):
    obs_mat[t][basis_idx] = t1[t]*(basis_prototypes[identities[basis_idx]][t- start_idxs[basis_idx]]);


x=[0.00000000e+00,   1.13579880e-03,  -9.79202654e-05,  -1.50851232e-03,
  -2.14445298e-03,   8.58891418e-04,  -8.13441741e-04,   9.59902998e-04,
   1.03570720e-03,   5.71705284e-04,   2.50564080e+00,   1.00960880e+00,
  -3.05237003e-01,  -1.40335472e+00,  -2.79434611e+00,  -4.33523351e+00,
  -5.23634537e+00,  -5.39350006e+00,  -4.98065806e+00,  -4.20678801e+00,
  -3.29258264e+00,  -2.20715409e+00,  -5.45734509e-01,   1.60061941e+00,
   2.94038308e+00,   3.26472512e+00,   3.20794043e+00,   2.63403743e+00,
   2.10859778e+00,   1.67313802e+00,   8.72048241e-01,  -3.00866292e-01,
  -1.10340175e+00,  -1.29493327e+00,  -1.50909672e+00,  -1.21605118e+00,
  -1.38942577e+00,  -1.74512755e+00,  -1.81617787e+00,  -2.11693658e+00,
  -1.98879203e+00,  -1.86813520e+00,  -1.72933632e+00,  -1.37532374e+00,
  -1.13344322e+00,  -6.91765448e-01,  -1.88214954e-01,   3.08390347e-01,
   7.06037853e-01,   1.02495917e+00,   1.04568180e+00,   1.12911921e+00,
   8.24700739e-01,   4.34859478e-01,  -4.17686285e-01,  -1.42860171e+00,
  -1.97238449e+00,  -1.79835783e+00,  -1.59737890e+00,  -1.01980567e+00,
  -4.69559687e-01,  -6.63993847e-01,  -5.63940623e-01,  -4.03142870e-01,
  -2.32601658e-01,   6.08060725e-02,   3.37414162e-01,   6.71796620e-01,
   1.05564526e+00,   1.57296801e+00,   2.10406024e+00,   2.96915033e+00,
   3.42193981e+00,   3.42001766e+00,   3.31654576e+00,   2.97972647e+00,
   2.67494878e+00,   2.59132530e+00,   2.09253072e+00,   1.68372803e+00,
   1.20184882e+00,   9.98726220e-01,   6.09429535e-01,   4.79960096e-01,
   3.84000493e-01,   9.50218558e-02,   1.16687729e-01,   1.89608028e-01,
   1.24052758e-01,   1.56899413e-01,   1.78849940e-01,   3.05907695e-01,
   4.50233028e-01,   7.11743316e-01,   1.06817949e+00,   1.40386026e+00,
   1.47616663e+00,   1.06371384e+00,   2.15963504e-01,  -8.10933029e-01,
  -1.81466857e+00,  -2.71823212e+00,  -3.90473117e+00,  -5.19509665e+00,
  -5.91295511e+00,  -5.83646176e+00,  -5.26951692e+00,  -4.17684190e+00,
  -3.01520363e+00,  -1.67989493e+00,  -1.56562373e-02,   1.96198591e+00,
   3.11675555e+00,   3.31961234e+00,   3.17332700e+00,   2.11055487e+00,
   1.30149225e+00,   7.53027208e-01,  -3.74226902e-01,  -1.90505100e+00,
  -2.87239592e+00,  -2.93042869e+00,  -2.86936869e+00,  -2.14914612e+00,
  -1.79789825e+00,  -1.60881074e+00,  -1.06144611e+00,  -5.52004613e-01,
  -2.65372861e-01,   9.32636057e-02,   1.51468741e-01,  -9.24667965e-02,
  -1.57810950e-01,  -6.68933148e-02,  -6.90147298e-02,  -2.59010250e-01,
  -3.40266584e-01,  -3.19935862e-01,  -2.49002309e-01,  -3.44622800e-02,
   1.11488167e-01,   2.44738205e-01,   4.38387455e-01,   7.54400769e-01,
   8.53371655e-01,   8.76798417e-01,   9.53346412e-01,   8.99195926e-01,
   8.95979724e-01,   8.99088058e-01,   8.92465728e-01,   8.33719019e-01,
   8.93433991e-01,   9.73522090e-01,   1.12034582e+00,   1.30470574e+00,
   1.43188042e+00,   1.37758629e+00,   1.26677791e+00,   1.19156577e+00]



tttime=time.time()

#tran_mat = np.eye(1)  #it's identity matrix,np.eye(dim_state,dim_state),delete later, no need
init_stt_mean = np.zeros(dim_state)
init_stt_cov = np.ones(dim_state)  

#prior_vars1 = np.ones((n_basis,), dtype=np.float) 
#prior_vars1[identities==0] = 1.0
#prior_vars1[identities==1] = 0.01
#prior_vars1[identities>1] = 0.0001
#prior_vars1 *= 5
prior_vars = [5.0, 0.05, 0.0005, 0.0005]

for n in range(dim_state):
  init_stt_cov[n] = prior_vars[identities[n]]



#tran_cov_mat = np.eye(1)#no need for this,delete later, transition noise is zero
#obs_mat = np.eye(cs_length,dim_state)  #whole observation matrix
obs_cov_mat = np.zeros(1)  # iid noise(change it later) 
obs_cov_mat[0]=0.000001
#observations = np.zeros(cs_length) 
observations = np.asarray(x)
fltr_stt_mean = np.zeros((cs_length, 2))  #now it becomes obs_mat
fltr_stt_cov = np.zeros((cs_length, 35*35))  #now it becomes pred_obs_mean & cov
logpdf = np.zeros((1,cs_length))

#since most nVidia devices only support single precision, transfer double to single

#tran_mat = tran_mat.astype(numpy.float32)
init_stt_mean = init_stt_mean.astype(numpy.float32)
init_stt_cov = init_stt_cov.astype(numpy.float32)
#tran_cov_mat = tran_cov_mat.astype(numpy.float32)
obs_mat = obs_mat.astype(numpy.float32)
obs_cov_mat = obs_cov_mat.astype(numpy.float32)
observations = observations.astype(numpy.float32)
fltr_stt_mean = fltr_stt_mean.astype(numpy.float32)
fltr_stt_cov = fltr_stt_cov.astype(numpy.float32)
logpdf = logpdf.astype(numpy.float32)

#tran_mat_gpu = cuda.mem_alloc(tran_mat.nbytes)
init_stt_mean_gpu = cuda.mem_alloc(init_stt_mean.nbytes)
init_stt_cov_gpu = cuda.mem_alloc(init_stt_cov.nbytes)
#tran_cov_mat_gpu = cuda.mem_alloc(tran_cov_mat.nbytes)
obs_mat_gpu = cuda.mem_alloc(obs_mat.nbytes)
obs_cov_mat_gpu = cuda.mem_alloc(obs_cov_mat.nbytes)
observations_gpu = cuda.mem_alloc(observations.nbytes)
fltr_stt_mean_gpu = cuda.mem_alloc(fltr_stt_mean.nbytes)
fltr_stt_cov_gpu = cuda.mem_alloc(fltr_stt_cov.nbytes)
logpdf_gpu = cuda.mem_alloc(logpdf.nbytes)

#cuda.memcpy_htod(tran_mat_gpu, tran_mat)
cuda.memcpy_htod(init_stt_mean_gpu, init_stt_mean)
cuda.memcpy_htod(init_stt_cov_gpu, init_stt_cov)
#cuda.memcpy_htod(tran_cov_mat_gpu, tran_cov_mat)
cuda.memcpy_htod(obs_mat_gpu, obs_mat)
cuda.memcpy_htod(obs_cov_mat_gpu, obs_cov_mat)
cuda.memcpy_htod(observations_gpu, observations)
cuda.memcpy_htod(fltr_stt_mean_gpu, fltr_stt_mean)
cuda.memcpy_htod(fltr_stt_cov_gpu, fltr_stt_cov)
cuda.memcpy_htod(logpdf_gpu,logpdf)
tttime1=time.time()
print cuda.mem_get_info()
mod = SourceModule("""
  #include <math.h>
  const float PI =  3.141592;
  __device__ void init(float *A, int A_II, int A_JJ){
    int n = A_II*A_JJ;
    for(int i = 0; i <n; i++){
      A[i] = 0;
    }
  }  

  __device__ void inverse(float *A, float *B, int II, int JJ){
    for(int i = 0; i < II; i++){
      for(int j = 0; j < JJ; j++){
        A[i+II*j] = B[i*JJ+j];
      }
    }
  }

  __device__ void matmul(float *A, float *B, float *temp, int II, int KK, int JJ){
    init(temp,II,JJ);
    for(int i = 0; i < II; i++){
      for(int j = 0; j < JJ; j++){
        for(int k = 0; k < KK; k++){
          temp[i*JJ+j] += A[i * KK + k] * B[k*JJ + j];
        }
      }
    }
  }

  __device__ void copyFromTemp(float *A, float *temp, int A_II, int A_KK){
    int n = A_II*A_KK;
    for(int i = 0; i <n; i++){
      A[i] = temp[i];
    }
  }

  __device__ void copyIntFromTemp(int *A, int *temp, int A_II, int A_KK){
    int n = A_II*A_KK;
    for(int i = 0; i <n; i++){
      A[i] = temp[i];
    }
  }
  __device__ void mat_add(float *A, float *B, float *temp, int II, int KK){
    int n = II * KK;
    init(temp,II,KK);
    for(int i = 0; i <n; i++){
      temp[i] = A[i] + B[i];
    }
  } 

  __device__ float log_pdf(float obs_t, float pred_obs_mean, float pred_obs_cov){
      if(pred_obs_cov==0){pred_obs_cov=0.000001;}
      float logpdf = log(1/sqrt(2*PI*pred_obs_cov)) + ((-0.5) * (obs_t - pred_obs_mean)*(obs_t - pred_obs_mean) / pred_obs_cov);
      return logpdf;
  }



  __device__ int active(float *obs_mat,float *active_obs_mat,int *active_index_new, float *pred_stt_mean,
                        float *pred_stt_cov,float *init_stt_mean,float *init_stt_cov,
                        int dim_state,int tid,int current_i, int iteration){
    int x = (tid + 1 - iteration + current_i)*dim_state;
    int count = 0;
    if(current_i == 0){
      for(int i = 0; i<dim_state; i++){
        if(obs_mat[x+i] != 0){
          active_obs_mat[count] = obs_mat[x+i]; 
          active_index_new[count] = i; 
          count++;
        }
      }
      init(pred_stt_cov,count,count);
      for(int j=0;j<count;j++){
        pred_stt_mean[j]=init_stt_mean[active_index_new[j]];
        pred_stt_cov[j*count+j] = init_stt_cov[active_index_new[j]];
      }
      return count;
    }else{
      for(int i = 0; i<dim_state; i++){
        if(obs_mat[x+i] != 0){
          active_obs_mat[count] = obs_mat[x+i]; 
          active_index_new[count] = i; 
          count++;                          
        }
      }
      return count;
    }
  }

  __device__ int isIndexActive(int index_new, int *active_index_old, int active_dim_state_old){
    for(int i=0; i<active_dim_state_old; i++){
      if(active_index_old[i]==index_new){return i;}
    }
    return -1;
  }
   
  __device__ void pred_stt_mean_cs_cal(float *pred_stt_mean, float *tran_mat, float *tran_cov_mat, 
                                      float *init_stt_cov,int *active_index_old,int *active_index_new,
                                      float *temp_fltr_stt_mean,int active_dim_state_old,int active_dim_state_new){
    int isActive;
    for (int i=0; i<active_dim_state_new; i++){
      isActive = isIndexActive(active_index_new[i],active_index_old,active_dim_state_old);
      if(isActive!= -1){
        pred_stt_mean[i] = temp_fltr_stt_mean[isActive];
        tran_mat[i]= isActive;
        tran_cov_mat[i]= 0;
      }else{
        pred_stt_mean[i]=0;
        tran_mat[i] = -1;
        tran_cov_mat[i] = init_stt_cov[active_index_new[i]];
      }
    }
  }


  __device__ void pred_stt_cov_cal(float *tran_mat,float *fltr_stt_cov,float *tran_cov_mat,
                                  float *pred_stt_cov,int active_dim_state_new, int active_dim_state_old,float *temp, float *temp1){
    init(temp,active_dim_state_new,active_dim_state_old);
    int m;
    for(int x=0; x<active_dim_state_old; x++){
      for(int y=0; y<active_dim_state_new; y++){
        m = tran_mat[y];
        if(m != -1){temp[y*active_dim_state_old+x]=fltr_stt_cov[m*active_dim_state_old+x];}
      }
    }
           
    inverse(temp1,temp,active_dim_state_new,active_dim_state_old);
    for(int x=0; x<active_dim_state_new; x++){
      for(int y=0; y<active_dim_state_new; y++){
        m = tran_mat[y];
        if(m != -1){
          pred_stt_cov[y*active_dim_state_new+x]=temp1[m*active_dim_state_new+x];
          }else{
          pred_stt_cov[y*active_dim_state_new+x]=0;
        }
      }
    }

    for(int i=0;i<active_dim_state_new;i++){
      pred_stt_cov[i*active_dim_state_new+i] += tran_cov_mat[i];
    }
  }

  __device__ float pred_obs_mean_cs_cal(float *active_obs_mat,float *pred_stt_mean,float pred_obs_mean,int active_dim_state,float *temp){
    matmul(active_obs_mat, pred_stt_mean, temp, 1, active_dim_state, 1);
    return temp[0];
  }

  __device__ float pred_obs_cov_cs_cal(float *active_obs_mat,float *pred_stt_cov,float *active_obs_mat_T,
                                      float *obs_cov_mat,float pred_obs_cov,int active_dim_state,float *temp,float *temp1){
    matmul(pred_stt_cov, active_obs_mat_T, temp, active_dim_state, active_dim_state, 1); 
    matmul(active_obs_mat, temp, temp1, 1, active_dim_state, 1);
    return temp1[0] + obs_cov_mat[0];
  }


  __device__ void matmul_kg(float *A, float *B, float *temp, int II, int KK, int JJ, float pred_obs_cov){
    if(pred_obs_cov==0){
      pred_obs_cov = 0.00001;
    }
    float x = 1/pred_obs_cov;
    init(temp,II,JJ); 
    for(int i = 0; i < II; i++){
      for(int j = 0; j < JJ; j++){
        for(int k = 0; k < KK; k++){
          temp[i*JJ+j] += A[i * KK + k] * B[k*JJ + j] * x;
        }
      }
    }
  }

  __device__ void kalman_gain_cs_cal(float *pred_stt_cov,float *active_obs_mat_T, float pred_obs_cov, float* kalman_gain, int active_dim_state){
    matmul_kg(pred_stt_cov, active_obs_mat_T, kalman_gain, active_dim_state, active_dim_state, 1, pred_obs_cov);
  }


 __device__ void fltr_stt_mean_cs_cal(float *pred_stt_mean,float *kalman_gain,
    float *observations,float pred_obs_mean,float *temp_fltr_stt_mean,int active_dim_state_new, int tid, int current_i, int iteration){
    float x;
    x = observations[tid + 1 - iteration + current_i] - pred_obs_mean;
    for (int i = 0; i < active_dim_state_new; i++){
      temp_fltr_stt_mean[i] = pred_stt_mean[i] + kalman_gain[i] * x;
    }
  }

  __device__ void fltr_stt_cov_cs_cal(float *pred_stt_cov,float *kalman_gain,
    float *active_obs_mat,float *temp_fltr_stt_cov,int active_dim_state_new, float *temp, float *temp1,int tid){
    matmul(kalman_gain, active_obs_mat, temp, active_dim_state_new, 1, active_dim_state_new);
    matmul(temp, pred_stt_cov, temp1, active_dim_state_new, active_dim_state_new, active_dim_state_new);
    for(int i=0; i <active_dim_state_new*active_dim_state_new; i++){
      temp_fltr_stt_cov[i] = pred_stt_cov[i] - temp1[i];
    }
  }



  __global__ void filter(
    float *init_stt_mean, 
    float *init_stt_cov,
    float *obs_mat,
    float *obs_cov_mat,
    float *observations,
    float *fltr_stt_mean,
    float *fltr_stt_cov,
    float *logpdf){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int iteration = 30;
    int dim_state = 168;
    float temp[35*35];
    float temp1[35*35];

    float pred_stt_mean[35];
    float pred_stt_cov[35*35];
    float pred_obs_mean;
    float pred_obs_cov;  
    float kalman_gain[35];
    float temp_fltr_stt_mean[35];
    float temp_fltr_stt_cov[35*35];
    float tran_mat[35];
    float tran_cov_mat[35];

    float active_obs_mat[35];

      int active_index_old[35];
      int active_index_new[35];
      int active_dim_state_old;
      int active_dim_state_new;


    if (tid < iteration){
      iteration = tid + 1;
    }

    for(int i = 0; i < iteration; i++){
      
      if (i ==0 ){
        init(active_obs_mat,35,1);
        active_dim_state_new = active(obs_mat,active_obs_mat,active_index_new,
                                      pred_stt_mean,pred_stt_cov,
                                      init_stt_mean,init_stt_cov,
                                      dim_state,tid,i,iteration);

      }else{
        active_dim_state_old = active_dim_state_new;
        copyIntFromTemp(active_index_old, active_index_new, 1, active_dim_state_old);
        init(active_obs_mat,35,1);
        active_dim_state_new = active(obs_mat,active_obs_mat,active_index_new,
                                      pred_stt_mean,pred_stt_cov,
                                      temp_fltr_stt_mean,temp_fltr_stt_cov,
                                      dim_state,tid,i,iteration);    

        init(tran_mat, active_dim_state_new, 1);
        init(tran_cov_mat, active_dim_state_new, 1);

        pred_stt_mean_cs_cal(pred_stt_mean, tran_mat, tran_cov_mat, init_stt_cov, active_index_old, active_index_new,
                                      temp_fltr_stt_mean, active_dim_state_old, active_dim_state_new);
        pred_stt_cov_cal(tran_mat, temp_fltr_stt_cov, 
                        tran_cov_mat, pred_stt_cov, active_dim_state_new,active_dim_state_old, temp, temp1);
      }
      
      pred_obs_mean = pred_obs_mean_cs_cal(active_obs_mat,pred_stt_mean,pred_obs_mean,active_dim_state_new,temp);
      pred_obs_cov = pred_obs_cov_cs_cal(active_obs_mat,
        pred_stt_cov,active_obs_mat,
        obs_cov_mat,pred_obs_cov,active_dim_state_new,temp,temp1);
      init(kalman_gain,35,1);
      kalman_gain_cs_cal(pred_stt_cov,
        active_obs_mat, pred_obs_cov, kalman_gain, active_dim_state_new);

 
      init(temp_fltr_stt_mean,35,1);
      fltr_stt_mean_cs_cal(pred_stt_mean,kalman_gain,
        observations, pred_obs_mean,temp_fltr_stt_mean,active_dim_state_new, tid, i, iteration);


      init(temp_fltr_stt_cov,35,35);
      fltr_stt_cov_cs_cal(pred_stt_cov,
        kalman_gain,active_obs_mat,
        temp_fltr_stt_cov, active_dim_state_new, temp, temp1, tid);
      }
      

      fltr_stt_mean[tid*2] = pred_obs_mean;
      fltr_stt_mean[tid*2+1] = pred_obs_cov;


      for(int i = 0; i < active_dim_state_new*active_dim_state_new; i++){
         fltr_stt_cov[tid*35*35+i] = temp_fltr_stt_mean[i];
       }
        


      logpdf[tid] = log_pdf(observations[tid],pred_obs_mean,pred_obs_cov);
    }
""")


start = time.time()

# for(int i = 0; i < dim_state; i++){
#         fltr_stt_cov[tid*dim_state+i] = temp_fltr_stt_cov[i];
#       }
# for(int i = 0; i < active_dim_state_new; i++){
#         fltr_stt_mean[tid*dim_state+i] = temp_fltr_stt_mean[i];
#       }

# fltr_stt_cov[tid*2] = pred_obs_mean;
# fltr_stt_cov[tid*2 +1] = pred_obs_cov;

filter = mod.get_function("filter")

start = time.time()

filter(init_stt_mean_gpu,
  init_stt_cov_gpu,
  obs_mat_gpu,
  obs_cov_mat_gpu,
  observations_gpu,
  fltr_stt_mean_gpu,
  fltr_stt_cov_gpu,
  logpdf_gpu,
  grid=(1,1), block=(cs_length,1,1))

context.synchronize()

end1 = time.time()

cuda.memcpy_dtoh(init_stt_mean, init_stt_mean_gpu)
cuda.memcpy_dtoh(init_stt_cov, init_stt_cov_gpu)
cuda.memcpy_dtoh(obs_mat, obs_mat_gpu)
cuda.memcpy_dtoh(obs_cov_mat, obs_cov_mat_gpu)
cuda.memcpy_dtoh(observations, observations_gpu)
cuda.memcpy_dtoh(fltr_stt_mean, fltr_stt_mean_gpu)
cuda.memcpy_dtoh(fltr_stt_cov, fltr_stt_cov_gpu)
cuda.memcpy_dtoh(logpdf, logpdf_gpu)

init_stt_mean_gpu.free()
init_stt_cov_gpu.free()
obs_mat_gpu.free()
obs_cov_mat_gpu.free()
observations_gpu.free()
fltr_stt_mean_gpu.free()
fltr_stt_cov_gpu.free()
logpdf_gpu.free()
print cuda.mem_get_info()

end2 = time.time()

elapsed = end2 - start
beforecopy = end1 - start
copy = end2-end1

#print fltr_stt_mean
#print fltr_stt_cov
print logpdf
print "marginal_likelihood:"
print np.sum(logpdf)
print "time(initialize + copyFromHostToDevice):"
print tttime1 - tttime
print "time(filter+marginal_likelihood):"
print beforecopy
print "time(copyFromDeviceToHost):"
print copy

# float pdf = 1/sqrt(2*PI*pred_obs_cov) * exp((-0.5) * (obs_t - pred_obs_mean)*(obs_t - pred_obs_mean) / pred_obs_cov);
#       return logpdf;





