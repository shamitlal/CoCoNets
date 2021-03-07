#include <torch/types.h>
using namespace torch;

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

// Cuda tensor accessor definitions
// restrict pointer traits piroritize speed over memory consumption
#define TensorAcc7R PackedTensorAccessor<scalar_t,7,RestrictPtrTraits,int32_t>
#define TensorAcc5R PackedTensorAccessor<scalar_t,5,RestrictPtrTraits,int32_t>
#define WITHIN_BOUNDS(x, y, z, D, H, W) (x >= 0 && x < D && y >= 0 && y < H && z >= 0 && z < W)
#define WITHIN_BOUNDS2(x, y, D, H) (x >= 0 && x < D && y >= 0 && y < H)

#define THREADS_FORWARD 32
#define THREADS_BACKWARD 5


namespace {
template <typename scalar_t>
__global__ void correlation_cuda_forward_kernel(
    const TensorAcc5R rInput1, // [D, H, W, N, C]
    const TensorAcc5R rInput2,
    TensorAcc7R output,
    int kD, int kH, int kW,
    int patchD, int patchH, int patchW,
    int padD, int padH, int padW,
    int dilation_patchD, int dilation_patchH, int dilation_patchW,
    int dD, int dH, int dW) {
  
  const int iD = rInput1.size(0);
  const int iH = rInput1.size(1);
  const int iW = rInput1.size(2);
  const int N = rInput1.size(3);
  const int C = rInput1.size(4);

  const int d = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int thread = threadIdx.x;

  const int start_i = -padD + d * dD;
  const int start_j = -padH + h * dH;
  const int start_k = -padW + w * dW;

  const int patchRadD = dilation_patchD * (patchD - 1) / 2;
  const int patchRadH = dilation_patchH * (patchH - 1) / 2;
  const int patchRadW = dilation_patchW * (patchW - 1) / 2;

  __shared__ scalar_t prod_sum[THREADS_FORWARD];

  for(int n = 0; n < N; ++n){
    for(int pd = 0; pd < patchD; ++pd){
      int pd_dilated = pd * dilation_patchD - patchRadD;
      for(int ph = 0; ph < patchH; ++ph){
	int ph_dilated = ph * dilation_patchH - patchRadH;
	for(int pw = 0; pw < patchW; ++pw){
	  int pw_dilated = pw * dilation_patchW - patchRadW;
	  prod_sum[thread] = 0;
	  for (int i=0; i<kD; ++i){
            int i1 = start_i + i;
            int i2 = i1 + pd_dilated;
	    if WITHIN_BOUNDS2(i1, i2, iD, iD){
	      for (int j=0; j<kH; ++j){
		int j1 = start_j + j;
		int j2 = j1 + ph_dilated;
		if WITHIN_BOUNDS2(j1, j2, iH, iH){
		  for (int k=0; k<kW; ++k){
		    int k1 = start_k + k;
		    int k2 = k1 + pw_dilated;
		    if WITHIN_BOUNDS2(k1, k2, iW, iW){
		      for (int c=thread; c<C; c += THREADS_FORWARD){
			scalar_t v1 = rInput1[i1][j1][k1][n][c];
			scalar_t v2 = rInput2[i2][j2][k2][n][c];
			prod_sum[thread] += v1 * v2;
		      }
		    }
		  }
		}
	      }
	    }
	  }
	      // accumulate 
	      __syncthreads();
	      if (thread == 0) {
		scalar_t reduce_sum = 0;
		for (int index = 0; index < THREADS_FORWARD; ++index) {
		  reduce_sum += prod_sum[index];
		}
		output[n][pd][ph][pw][d][h][w] = reduce_sum;
	      }
	}
      }
    }
  }
}


template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input1(
    const TensorAcc7R gradOutput, // [N, pH, pW, pD, H, W, D]
    const TensorAcc5R input2, // [D, H, W, N, C]
    TensorAcc5R gradInput1,
    int kD, int kH, int kW,
    int patchD, int patchH, int patchW,
    int padD, int padH, int padW,
    int dilation_patchD, int dilation_patchH, int dilation_patchW,
    int dD, int dH, int dW,
    int batch) {

  const int iD = input2.size(2);
  const int iH = input2.size(3);
  const int iW = input2.size(4);

  const int D = gradOutput.size(4);
  const int H = gradOutput.size(5);
  const int W = gradOutput.size(6);

  const int patchRadD = (patchD - 1) / 2;
  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  
  const int n = batch;
  const int C = input2.size(1);
  const int d = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int pd_off = threadIdx.x;
  const int ph_off = threadIdx.y;
  const int pw_off = threadIdx.z;

  const int d_2 = d + padD;
  const int h_2 = h + padH;
  const int w_2 = w + padW;
  const int start_i2 = d_2 / dD;
  const int start_j2 = h_2 / dH;
  const int start_k2 = w_2 / dW;
  /*we perform a module but since we have the quotient, we
  can cheat a bit*/
  const int d_off = d_2 - start_i2 * dD;
  const int h_off = h_2 - start_j2 * dH;
  const int w_off = w_2 - start_k2 * dW;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD][THREADS_BACKWARD];
  for(int c = 0; c < C; c++){
  prod_sum[pd_off][ph_off][pw_off] = 0;
    for (int pd = pd_off; pd < patchD; pd += THREADS_BACKWARD){
      int i1 = d + dilation_patchD * (pd - patchRadD);
      for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
        int j1 = h + dilation_patchH * (ph - patchRadH);
        for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
          int k1 = w + dilation_patchW * (pw - patchRadW);
          if WITHIN_BOUNDS(i1, j1, k1, iD, iH, iW) {
            scalar_t val = input2[n][c][i1][j1][k1];
            for(int tmp1 = d_off, i = 0; tmp1 < kD; tmp1 += dD, ++i) {
              int i2 = start_i2 - i;
              for(int tmp2 = h_off, j = 0; tmp2 < kH; tmp2 += dH, ++j) {
                int j2 = start_j2 - j;
                for(int tmp3 = w_off, k = 0; tmp3 < kW; tmp3 += dW, ++k) {
                  int k2 = start_k2 - k;
                  if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
                    prod_sum[pd_off][ph_off][pw_off] += gradOutput[n][pd][ph][pw][i2][j2][k2] * val;
            
                }
	      }
            }
          }
        }
      }
    }

    __syncthreads();

    if (pd_off == 0 && ph_off == 0 && pw_off == 0){
      scalar_t reduce_sum =0;
      for (int pd = 0; pd < THREADS_BACKWARD; ++pd){
        for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
          for (int pw = 0; pw < THREADS_BACKWARD; ++pw){
            reduce_sum += prod_sum[pd][ph][pw];
          }
        }
      }
      gradInput1[n][c][d][h][w] = reduce_sum;
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input2(
    const TensorAcc7R gradOutput, // [N, pH, pW, pD, H, W, D]
    const TensorAcc5R input1, // [D, H, W, N, C]
    TensorAcc5R gradInput2,
    int kD, int kH, int kW,
    int patchD, int patchH, int patchW,
    int padD, int padH, int padW,
    int dilation_patchD, int dilation_patchH, int dilation_patchW,
    int dD, int dH, int dW,
    int batch) {

  const int iD = input1.size(2);
  const int iH = input1.size(3);
  const int iW = input1.size(4);

  const int D = gradOutput.size(4);
  const int H = gradOutput.size(5);
  const int W = gradOutput.size(6);

  const int patchRadD = (patchD - 1) / 2;
  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  
  const int n = batch;
  const int C = input1.size(1);
  const int d = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int pd_off = threadIdx.x;
  const int ph_off = threadIdx.y;
  const int pw_off = threadIdx.z;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD][THREADS_BACKWARD];

  for(int c = 0; c < C; c++){
  prod_sum[pd_off][ph_off][pw_off] = 0;
    for (int pd = pd_off; pd < patchD; pd += THREADS_BACKWARD){
      int i1 = d + dilation_patchD * (pd - patchRadD);
      for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
        int j1 = h + dilation_patchH * (ph - patchRadH);
        for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
          int k1 = w + dilation_patchW * (pw - patchRadW);
          if WITHIN_BOUNDS(i1, j1, k1, iD, iH, iW) {
            scalar_t val = input1[n][c][i1][j1][k1];
            const int d_2 = d + padD;
            const int h_2 = h + padH;
	    const int w_2 = w + padW;
            const int start_i2 = d_2 / dD;
       	    const int start_j2 = h_2 / dH;
	    const int start_k2 = w_2 / dW;
	    /*we perform a module but since we have the quotient, we
	    can cheat a bit*/
	    const int d_off = d_2 - start_i2 * dD;
	    const int h_off = h_2 - start_j2 * dH;
	    const int w_off = w_2 - start_k2 * dW;


            for(int tmp1 = d_off, i = 0; tmp1 < kD; tmp1 += dD, ++i) {
              int i2 = start_i2 - i;
              for(int tmp2 = h_off, j = 0; tmp2 < kH; tmp2 += dH, ++j) {
                int j2 = start_j2 - j;
                for(int tmp3 = w_off, k = 0; tmp3 < kW; tmp3 += dW, ++k) {
                  int k2 = start_k2 - k;
                  if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
                    prod_sum[pd_off][ph_off][pw_off] += gradOutput[n][pd][ph][pw][i2][j2][k2] * val;
            
                }
	      }
            }
          }
        }
      }
    }

    __syncthreads();

    if (pd_off == 0 && ph_off == 0 && pw_off == 0){
      scalar_t reduce_sum =0;
      for (int pd = 0; pd < THREADS_BACKWARD; ++pd){
        for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
          for (int pw = 0; pw < THREADS_BACKWARD; ++pw){
            reduce_sum += prod_sum[pd][ph][pw];
          }
        }
      }
      gradInput2[n][c][d][h][w] = reduce_sum;
    }
    __syncthreads();
  }
}
}

/*
template <typename scalar_t>
__global__ void correlation_cuda_backward_kernel_input2(
    const TensorAcc5R gradOutput,
    const TensorAcc4R input1,
    TensorAcc4R gradInput2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW,
    int batch) {
  const int iH = input1.size(2);
  const int iW = input1.size(3);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int H = gradOutput.size(3);
  const int W = gradOutput.size(4);
  
  const int n = batch;
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int ph_off = threadIdx.x;
  const int pw_off = threadIdx.y;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
  prod_sum[ph_off][pw_off] = 0;

  for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
    int i1 = h - dilation_patchH * (ph - patchRadH);
    for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
      int j1 = w - dilation_patchW * (pw - patchRadW);
      if WITHIN_BOUNDS(i1, j1, iH, iW) {
        scalar_t val = input1[n][c][i1][j1];
        
        const int h_2 = i1 + padH;
        const int w_2 = j1 + padW;
        const int start_i2 = h_2 / dH;
        const int start_j2 = w_2 / dW;
        const int h_off = h_2 - start_i2 * dH;
        const int w_off = w_2 - start_j2 * dW;
        
        for(int tmp1 = h_off, i = 0; tmp1 < kH; tmp1 += dH, ++i) {
          int i2 = start_i2 - i;
          for(int tmp2 = w_off, j = 0; tmp2 < kW; tmp2 += dW, ++j) {
            int j2 = start_j2 - j;
            if WITHIN_BOUNDS(i2, j2, H, W) {
              prod_sum[ph_off][pw_off] += gradOutput[n][ph][pw][i2][j2] * val;
            }
          }
        }
      }
    }
  }

  __syncthreads();

  if (ph_off == 0 && pw_off == 0){
    scalar_t reduce_sum =0;
    for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
      for (int pw = 0; pw < THREADS_BACKWARD; ++pw){
        reduce_sum += prod_sum[ph][pw];
      }
    }
    gradInput2[n][c][h][w] = reduce_sum;
  }
}
}
*/
torch::Tensor correlation_cuda_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kD, int kH, int kW,
    int patchD, int patchH, int patchW,
    int padD, int padH, int padW,
    int dilation_patchD, int dilation_patchH, int dilation_patchW,
    int dD, int dH, int dW) {
  
  const int batch_size = input1.size(0);
  const int iD = input1.size(2);
  const int iH = input1.size(3);
  const int iW = input1.size(4);

  const auto oD = (iD + 2 * padD - kD) / dD + 1;
  const auto oH = (iH + 2 * padH - kH) / dH + 1;
  const auto oW = (iW + 2 * padW - kW) / dW + 1;
  auto output = torch::zeros({batch_size, patchD, patchH, patchW, oD, oH, oW}, input1.options());
  
  auto trInput1 = input1.permute({2, 3, 4, 0, 1}).contiguous();
  auto trInput2 = input2.permute({2, 3, 4, 0, 1}).contiguous();
  
  const int threads = THREADS_FORWARD;
  const dim3 blocks(oD, oH, oW);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_forward_cuda", ([&] {
    TensorAcc5R trInput1_acc = trInput1.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc5R trInput2_acc = trInput2.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc7R output_acc = output.packed_accessor<scalar_t,7,RestrictPtrTraits,int32_t>();
    correlation_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        trInput1_acc, trInput2_acc, output_acc,
        kD, kH, kW, 
	patchD, patchH, patchW, 
	padD, padH, padW,
        dilation_patchD, dilation_patchH, dilation_patchW, 
	dD, dH, dW);
  }));

  return output;
}

std::vector<torch::Tensor> correlation_cuda_backward(
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor gradOutput,
    int kD, int kH, int kW,
    int patchD, int patchH, int patchW,
    int padD, int padH, int padW,
    int dilation_patchD, int dilation_patchH, int dilation_patchW,
    int dD, int dH, int dW) {
  
  auto gradInput1 = torch::zeros_like(input1);
  auto gradInput2 = torch::zeros_like(input2);

  const int batch_size = input1.size(0);
  const int iD = input1.size(2);
  const int iH = input1.size(3);
  const int iW = input1.size(4);
  const int C = input1.size(1);

  const dim3 blocks(iD, iH, iW);
  const dim3 threads(THREADS_BACKWARD, THREADS_BACKWARD, THREADS_BACKWARD);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_backward_cuda", ([&] {
    TensorAcc5R input1_acc = input1.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc5R input2_acc = input2.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc5R gradInput1_acc = gradInput1.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc5R gradInput2_acc = gradInput2.packed_accessor<scalar_t,5,RestrictPtrTraits,int32_t>();
    TensorAcc7R gradOutput_acc = gradOutput.packed_accessor<scalar_t,7,RestrictPtrTraits,int32_t>();


    for (int n = 0; n < batch_size; ++n){
      correlation_cuda_backward_kernel_input1<scalar_t><<<blocks, threads>>>(
          gradOutput_acc, input2_acc, gradInput1_acc,
          kD, kH, kW, patchD, patchH, patchW, padD, padH, padW,
          dilation_patchD, dilation_patchH, dilation_patchW, dD, dH, dW,
          n);
    }

    for (int n = 0; n < batch_size; ++n){
      correlation_cuda_backward_kernel_input2<scalar_t><<<blocks, threads>>>(
          gradOutput_acc, input1_acc, gradInput2_acc,
          kD, kH, kW, patchD, patchH, patchW, padD, padH, padW,
          dilation_patchD, dilation_patchH, dilation_patchW, dD, dH, dW,
          n);
    }
  }));

  return {gradInput1, gradInput2};
}
