extern "C"{
	#include "render.h"
	#include "stdio.h"
}

#define BLOCK_SIZE 512

__global__ void heatmap_render_saturated(unsigned char* colorscheme, size_t ncolors, float saturation, int width, int height, float* bufarray, unsigned char* colorbuf)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < width && y < height){
		int idx = width*y+x;
		float val = (bufarray[idx] > saturation ? saturation : bufarray[idx])/saturation;
		size_t idc = (size_t)((float)(ncolors-1)*val + 0.5f);

		if(val >= 0.0f && idc < ncolors){
			for(int i=0;i<4;i++){
				colorbuf[idx*4+i] = colorscheme[idc*4 + i];
			}
		}
	}
}

void heatmap_render_saturated_to_gpu(const heatmap_t* h, const heatmap_colorscheme_t* colorscheme, float saturation, unsigned char* colorbuf) 
{
	// Initialize device pointer for colorcheme
	size_t d_colorscheme_size = 4 * colorscheme->ncolors * sizeof(unsigned char);

	unsigned char* colors = (unsigned char*)malloc(d_colorscheme_size);
	memcpy(colors, colorscheme->colors, d_colorscheme_size);

	size_t ncolors = colorscheme->ncolors;
	unsigned char* d_colors;
	
	cudaMalloc(&d_colors, d_colorscheme_size);
	cudaMemcpy(d_colors, colors, d_colorscheme_size, cudaMemcpyHostToDevice);

	float* d_bufarray;
	size_t d_bufarray_size = h->h * h->w * sizeof(float);
	cudaMalloc(&d_bufarray, d_bufarray_size);
	cudaMemcpy(d_bufarray, h->buf, d_bufarray_size, cudaMemcpyHostToDevice);

	// Initialize device pointer for colorbuf
	unsigned char* d_colorbuf;
	size_t d_colorbuf_size = h->w*h->h*4 * sizeof(unsigned char);
	cudaMalloc(&d_colorbuf, d_colorbuf_size);

	// Define grid and block sizing
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid(h->w / dim_block.x, h->h / dim_block.y);

	// Invoke kernel
	heatmap_render_saturated<<<dim_grid, dim_block>>>(d_colors, ncolors, saturation, h->w, h->h, d_bufarray, d_colorbuf);
	cudaDeviceSynchronize();

	// printf("copying result ... \n");
	cudaMemcpy(colorbuf, d_colorbuf, d_colorbuf_size, cudaMemcpyDeviceToHost);

	cudaFree(d_bufarray);
    cudaFree(d_colors);
    cudaFree(d_colorbuf);
    free(colors);
}