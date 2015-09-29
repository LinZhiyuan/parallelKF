"""when delete init in kalman_gain_ar_cal, it runs even slower"""
from pycuda.autoinit import context
import pycuda.driver as cuda
import numpy as np 
from pycuda.compiler import SourceModule
import numpy
import time
from pycuda.driver import initialize_profiler


print cuda.mem_get_info()




tttime=time.time()
total_time = 128
dim_state_ar = 10
dim_obs = 1



tran_mat_ar_const = np.zeros(dim_state_ar*dim_state_ar) #constant
init_stt_mean_ar_const = np.zeros((1, dim_state_ar)) #constant
init_stt_cov_ar_const = 0.1*np.eye(dim_state_ar) #constant
tran_cov_mat_ar = np.zeros((1,1))
tran_cov_mat_ar[0][0]=0.1

fltr_stt_mean_ar = np.zeros((total_time, dim_state_ar))
fltr_stt_cov_ar = np.zeros((total_time,dim_state_ar,dim_state_ar))
logpdf = np.zeros((1,total_time))


pred_stt_mean_ar = np.zeros((total_time, dim_state_ar));
pred_stt_cov_ar = np.zeros((total_time,dim_state_ar,dim_state_ar));
pred_obs_mean_ar = np.zeros(total_time);
pred_obs_cov_ar = np.zeros(total_time);
kalman_gain_ar = np.zeros((total_time, dim_state_ar));

tran_mat_ar_const[0]=  0.3
tran_mat_ar_const[1]= -0.1
tran_mat_ar_const[2]=  0.2
tran_mat_ar_const[3]=  0.1
tran_mat_ar_const[4]= -0.05
tran_mat_ar_const[5]=  0.1
tran_mat_ar_const[6]= -0.05
tran_mat_ar_const[7]=  0.03
tran_mat_ar_const[8]= -0.01
tran_mat_ar_const[9]=  0.01
for i in range(1,dim_state_ar):
  tran_mat_ar_const[i*dim_state_ar+i-1]=1;

x= np.linspace(0, 0.8, total_time)
for i in range(4):
    x[20*i+20] = np.NAN
#for i in range(25):
#    x[4*i+4] = np.nan

observations_ar_const = np.array(x) #constant

#since most nVidia devices only support single precision, transfer double to single


tran_mat_ar_const = tran_mat_ar_const.astype(numpy.float32)
init_stt_mean_ar_const = init_stt_mean_ar_const.astype(numpy.float32)
init_stt_cov_ar_const = init_stt_cov_ar_const.astype(numpy.float32)
tran_cov_mat_ar = tran_cov_mat_ar.astype(numpy.float32)
#obs_mat_ar = obs_mat_ar.astype(numpy.float32)
#obs_cov_mat_ar = obs_cov_mat_ar.astype(numpy.float32)
observations_ar_const = observations_ar_const.astype(numpy.float32)
fltr_stt_mean_ar = fltr_stt_mean_ar.astype(numpy.float32)
fltr_stt_cov_ar = fltr_stt_cov_ar.astype(numpy.float32)
logpdf = logpdf.astype(numpy.float32)
pred_stt_mean_ar = pred_stt_mean_ar.astype(numpy.float32)
pred_stt_cov_ar = pred_stt_cov_ar.astype(numpy.float32)
pred_obs_mean_ar = pred_obs_mean_ar.astype(numpy.float32)
pred_obs_cov_ar = pred_obs_cov_ar.astype(numpy.float32)
kalman_gain_ar = kalman_gain_ar.astype(numpy.float32)

#tran_mat_ar_gpu = cuda.mem_alloc(tran_mat_ar.nbytes)
#init_stt_mean_ar_gpu = cuda.mem_alloc(init_stt_mean_ar_const.nbytes)
#init_stt_cov_ar_gpu = cuda.mem_alloc(init_stt_cov_ar.nbytes)
tran_cov_mat_ar_gpu = cuda.mem_alloc(tran_cov_mat_ar.nbytes)
#obs_mat_ar_gpu = cuda.mem_alloc(obs_mat_ar.nbytes)
#obs_cov_mat_ar_gpu = cuda.mem_alloc(obs_cov_mat_ar.nbytes)
#observations_ar_gpu = cuda.mem_alloc(observations_ar.nbytes)
fltr_stt_mean_ar_gpu = cuda.mem_alloc(fltr_stt_mean_ar.nbytes)
fltr_stt_cov_ar_gpu = cuda.mem_alloc(fltr_stt_cov_ar.nbytes)
logpdf_gpu = cuda.mem_alloc(logpdf.nbytes)
pred_stt_mean_ar_gpu = cuda.mem_alloc(pred_stt_mean_ar.nbytes)
pred_stt_cov_ar_gpu = cuda.mem_alloc(pred_stt_cov_ar.nbytes)
pred_obs_mean_ar_gpu = cuda.mem_alloc(pred_obs_mean_ar.nbytes)
pred_obs_cov_ar_gpu = cuda.mem_alloc(pred_obs_cov_ar.nbytes)
kalman_gain_ar_gpu = cuda.mem_alloc(kalman_gain_ar.nbytes)


#cuda.memcpy_htod(tran_mat_ar_gpu, tran_mat_ar)
#cuda.memcpy_htod(init_stt_mean_ar_gpu, init_stt_mean_ar)
#cuda.memcpy_htod(init_stt_cov_ar_gpu, init_stt_cov_ar)
cuda.memcpy_htod(tran_cov_mat_ar_gpu, tran_cov_mat_ar)
#cuda.memcpy_htod(obs_mat_ar_gpu, obs_mat_ar)
#cuda.memcpy_htod(obs_cov_mat_ar_gpu, obs_cov_mat_ar)
#cuda.memcpy_htod(observations_ar_gpu, observations_ar)
cuda.memcpy_htod(fltr_stt_mean_ar_gpu, fltr_stt_mean_ar)
cuda.memcpy_htod(fltr_stt_cov_ar_gpu, fltr_stt_cov_ar)
cuda.memcpy_htod(logpdf_gpu,logpdf)
cuda.memcpy_htod(pred_stt_mean_ar_gpu,pred_stt_mean_ar)
cuda.memcpy_htod(pred_stt_cov_ar_gpu,pred_stt_cov_ar)
cuda.memcpy_htod(pred_obs_mean_ar_gpu,pred_obs_mean_ar)
cuda.memcpy_htod(pred_obs_cov_ar_gpu,pred_obs_cov_ar)
cuda.memcpy_htod(kalman_gain_ar_gpu,kalman_gain_ar)

tttime1=time.time()

mod = SourceModule("""
  #include <math.h>
  __device__ __constant__ float init_stt_mean_ar[10];
  __device__ __constant__ float init_stt_cov_ar[100];
  __device__ __constant__ float tran_mat_ar[100];
  __device__ __constant__ float observations_ar[4096];

  const float PI =  3.141592;
  const int WARP = 32;
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
  
  __device__ void matmul_offset1(float *A, float *B, float *temp, int II, int KK, int JJ){
    init(temp,II,JJ);
    for(int i = 0; i < II; i++){
      for(int j = 0; j < JJ; j++){
        for(int k = 0; k < KK; k++){
          temp[i*JJ+j] += A[i * KK + k] * B[(k*JJ + j)*WARP];
        }
      }
    }
  }

 __device__ void copyFromTemp_offset(float *A, int A_II, int A_KK){
    int n = A_II*A_KK;
    for(int i = 0; i <n; i++){
      A[WARP*i] = init_stt_cov_ar[i];
    }
  }

  __device__ void copyFromTemp(float *A, int A_II, int A_KK){
    int n = A_II*A_KK;
    for(int i = 0; i <n; i++){
      A[i] = init_stt_mean_ar[i];
    }
  }
  __device__ void mat_add(float *A, float *B, float *temp, int II, int KK){
    int n = II * KK;
    init(temp,II,KK);
    for(int i = 0; i <n; i++){
      temp[i] = A[i] + B[i];
    }
  } 

  __device__ float log_pdf(float pred_obs_mean_ar, float pred_obs_cov_ar,int tid){
	float logpdf =0;
	if(observations_ar[tid] >= -99999999 && observations_ar[tid]<=999999999){
	logpdf = log(1/sqrt(2*PI*pred_obs_cov_ar)) + ((-0.5) * (observations_ar[tid] - pred_obs_mean_ar)*(observations_ar[tid] - pred_obs_mean_ar) / pred_obs_cov_ar);
	return pred_obs_mean_ar;}
	return 0;
  }





  __device__ void pred_stt_mean_ar_cal(float *fltr_stt_mean_ar,float *pred_stt_mean_ar,int dim_state_ar,float *temp){
    for(int i=1;i<dim_state_ar;i++){pred_stt_mean_ar[i]=fltr_stt_mean_ar[(i-1)*WARP];}
    temp[0]=0;
    for(int i=0;i<dim_state_ar;i++){
      temp[0]+= tran_mat_ar[i]*fltr_stt_mean_ar[WARP*i];
    }
    pred_stt_mean_ar[0] = temp[0];
  }

 __device__ void pred_stt_cov_cs_cal(float *fltr_stt_cov,float *tran_cov_mat_ar,float *pred_stt_cov,int dim_state_ar,float *temp,float *temp1){
    
    init(temp,dim_state_ar,dim_state_ar);
    for(int x=0; x<dim_state_ar; x++){
      for(int y=0;y<dim_state_ar;y++){
        if(y==0){
          for(int ar=0;ar<dim_state_ar;ar++){
            temp[x]+=tran_mat_ar[ar]*fltr_stt_cov[((ar)*dim_state_ar+x)*WARP];}
        }else{temp[y*dim_state_ar+x]=fltr_stt_cov[((y-1)*dim_state_ar+x)*WARP];}
      }
    }
           
    inverse(temp1,temp,dim_state_ar,dim_state_ar);
    for(int x=0; x<dim_state_ar; x++){
      for(int y=0;y<dim_state_ar;y++){
        if(y==0){
          for(int ar=0;ar<dim_state_ar;ar++){
            if(ar == 0){pred_stt_cov[x*WARP]=0;}
            pred_stt_cov[x*WARP]+=tran_mat_ar[ar]*temp1[ar*dim_state_ar+x];}
        }else{pred_stt_cov[(y*dim_state_ar+x)*WARP]=temp1[(y-1)*dim_state_ar+x];}
      }
    }
    pred_stt_cov[0] += tran_cov_mat_ar[0];
  }


  __device__ void kalman_gain_ar_cal(float *pred_stt_cov_ar, float pred_obs_cov_ar, float* kalman_gain_ar, int dim_state_ar){
    if(pred_obs_cov_ar==0){
      pred_obs_cov_ar = 0.000001;
    }
    const float x = 1/pred_obs_cov_ar;
    for(int i = 0; i < dim_state_ar; i++){
          kalman_gain_ar[i] = pred_stt_cov_ar[(i * dim_state_ar)*WARP] * x;
    }
  }


  __device__ void fltr_stt_mean_ar_cal(float *pred_stt_mean_ar,float *kalman_gain_ar,
    float pred_obs_mean_ar,float *temp_fltr_stt_mean_ar,int dim_state_ar, int tid, int current_i, int iteration){
    const float x = observations_ar[tid + 1 - iteration + current_i] - pred_obs_mean_ar;
    for (int i = 0; i < dim_state_ar; i++){
      temp_fltr_stt_mean_ar[WARP*i] = pred_stt_mean_ar[i] + kalman_gain_ar[i] * x;
    }
  }

  __device__ void fltr_stt_cov_ar_cal(float *pred_stt_cov,float *kalman_gain_ar,
    float *temp_fltr_stt_cov, int dim_state, float *temp, float *temp1, int tid){

    matmul_offset1(kalman_gain_ar, pred_stt_cov, temp1, dim_state, 1, dim_state);

    for(int i=0; i <dim_state*dim_state; i++){
      temp_fltr_stt_cov[i*WARP] = pred_stt_cov[i*WARP] - temp1[i];
    }
  }




  __global__ void filter(
    float *tran_cov_mat_ar,
    float *fltr_stt_mean_ar,
    float *fltr_stt_cov_ar,
    float *logpdf,
    float *pred_obs_mean_ar,
    float *pred_obs_cov_ar,
    float *pred_stt_cov_ar){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int w = tid / WARP;
    int wid = tid % WARP;
    int offset_mean =w * WARP * 10;
    int offset_cov = w * WARP * 100;
    int iteration = 11;
    int dim_state_ar = 10;
    float temp[100];
    float temp1[100];

    float kalman_gain_ar[10];
    float pred_stt_mean_ar[10];

  
    if (tid < iteration){
      iteration = tid + 1;
    }else{iteration=iteration;}

    for(int i = 0; i < iteration; i++){

      if (i ==0 ){
        copyFromTemp(pred_stt_mean_ar, dim_state_ar, 1);
        copyFromTemp_offset(&pred_stt_cov_ar[offset_cov+wid], dim_state_ar, dim_state_ar);
      }else{
        pred_stt_mean_ar_cal(&fltr_stt_mean_ar[offset_mean+wid], pred_stt_mean_ar, dim_state_ar, temp);
        pred_stt_cov_cs_cal(&fltr_stt_cov_ar[offset_cov+wid],
          tran_cov_mat_ar, &pred_stt_cov_ar[offset_cov+wid], dim_state_ar, temp,temp1);
      }
      if(!(observations_ar[tid + 1 - iteration + i]>=-99999999 && observations_ar[tid + 1 - iteration + i]<=99999999999)){i=i+1;}


      pred_obs_mean_ar[tid] = pred_stt_mean_ar[0];
      pred_obs_cov_ar[tid] = pred_stt_cov_ar[offset_cov+wid];

      kalman_gain_ar_cal(&pred_stt_cov_ar[offset_cov+wid],
        pred_obs_cov_ar[tid], kalman_gain_ar, dim_state_ar);

      fltr_stt_mean_ar_cal(pred_stt_mean_ar, kalman_gain_ar,
        pred_obs_mean_ar[tid], &fltr_stt_mean_ar[offset_mean+wid], dim_state_ar, tid, i, iteration);

      fltr_stt_cov_ar_cal(&pred_stt_cov_ar[offset_cov+wid],
        kalman_gain_ar, 
        &fltr_stt_cov_ar[offset_cov+wid], dim_state_ar, temp, temp1, tid);
      }
      
      logpdf[tid] = log_pdf(pred_obs_mean_ar[tid], pred_obs_cov_ar[tid],tid);
    }
""")


start = time.time()


context.set_cache_config(cuda.func_cache.PREFER_L1)

filter = mod.get_function("filter")
init_stt_mean_ar =  mod.get_global('init_stt_mean_ar')[0] 
init_stt_cov_ar =  mod.get_global('init_stt_cov_ar')[0] 
tran_mat_ar =  mod.get_global('tran_mat_ar')[0] 
observations_ar =  mod.get_global('observations_ar')[0] 

cuda.memcpy_htod(init_stt_mean_ar, init_stt_mean_ar_const)
cuda.memcpy_htod(init_stt_cov_ar, init_stt_cov_ar_const)
cuda.memcpy_htod(tran_mat_ar, tran_mat_ar_const)
cuda.memcpy_htod(observations_ar, observations_ar_const)



start = time.time()

filter(tran_cov_mat_ar_gpu,
  fltr_stt_mean_ar_gpu,
  fltr_stt_cov_ar_gpu,
  logpdf_gpu,
  pred_obs_mean_ar_gpu,
  pred_obs_cov_ar_gpu,
  pred_stt_cov_ar_gpu,
  grid=(8,1), block=(total_time/8,1,1))

context.synchronize()

end1 = time.time()

#cuda.memcpy_dtoh(tran_mat_ar, tran_mat_ar_gpu)
#cuda.memcpy_dtoh(init_stt_mean_ar, init_stt_mean_ar_gpu)
#cuda.memcpy_dtoh(init_stt_cov_ar, init_stt_cov_ar_gpu)
cuda.memcpy_dtoh(tran_cov_mat_ar, tran_cov_mat_ar_gpu)
#cuda.memcpy_dtoh(obs_mat_ar, obs_mat_ar_gpu)
#cuda.memcpy_dtoh(obs_cov_mat_ar, obs_cov_mat_ar_gpu)
#cuda.memcpy_dtoh(observations_ar, observations_ar_gpu)
cuda.memcpy_dtoh(fltr_stt_mean_ar, fltr_stt_mean_ar_gpu)
cuda.memcpy_dtoh(fltr_stt_cov_ar, fltr_stt_cov_ar_gpu)
cuda.memcpy_dtoh(logpdf, logpdf_gpu)
cuda.memcpy_dtoh(kalman_gain_ar, kalman_gain_ar_gpu)

end2 = time.time()

elapsed = end2 - start
beforecopy = end1 - start
copy = end2-end1
print cuda.mem_get_info()

#print fltr_stt_mean_ar
#print fltr_stt_cov_ar
print logpdf
print "marginal_likelihood:"
print np.sum(logpdf)
print "time(initialize + copyFromHostToDevice):"
print tttime1 - tttime
print "time(filter+marginal_likelihood):"
print beforecopy
print "time(copyFromDeviceToHost):"
print copy

# float pdf = 1/sqrt(2*PI*pred_obs_cov_ar) * exp((-0.5) * (obs_t - pred_obs_mean_ar)*(obs_t - pred_obs_mean_ar) / pred_obs_cov_ar);
#       return logpdf;





