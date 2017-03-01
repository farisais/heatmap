extern "C"{
	#include "render.h"
}

#define BLOCK_SIZE 512

__global__ void heatmap_render_saturated(heatmap_colorscheme_t* colorscheme, float saturation, int width, float* bufarray, unsigned char* colorbuf)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = width*y+x;

	float val = (bufarray[idx] > saturation ? saturation : bufarray[idx])/saturation;
	size_t idc = (size_t)((float)(colorscheme->ncolors-1)*val + 0.5f);
	
	for(int i=0;i<4;i++){
		colorbuf[idx*4+i] = colorscheme->colors[idc*4 + i];
	}
}

void heatmap_render_saturated_to_gpu(const heatmap_t* h, const heatmap_colorscheme_t* colorscheme, float saturation, unsigned char* colorbuf) 
{
	// Initialize device pointer for colorcheme
	heatmap_colorscheme_t d_colorscheme;
	size_t d_colorscheme_size = 4 * colorscheme->ncolors;
	d_colorscheme.ncolors = colorscheme->ncolors;
	cudaMalloc((void **)&d_colorscheme.colors, d_colorscheme_size);
	cudaMemcpy((char*)d_colorscheme.colors, (char *)colorscheme->colors, d_colorscheme_size, cudaMemcpyHostToDevice);

	// Initialize device pointer for bufarray
	float* d_bufarray;
	size_t d_bufarray_size = h->h * h->w;
	cudaMalloc((void **)&d_bufarray, sizeof(float) * d_bufarray_size);
	cudaMemcpy(d_bufarray, h->buf, d_bufarray_size, cudaMemcpyHostToDevice);

	// Initialize device pointer for colorbuf
	unsigned char* d_colorbuf;
	size_t d_colorbuf_size = h->w*h->h*4;
	cudaMalloc((void **)&d_colorbuf, sizeof(unsigned char) * d_colorbuf_size);
	cudaMemcpy(d_colorbuf, colorbuf, d_colorbuf_size, cudaMemcpyHostToDevice);

	// Define grid and block sizing
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid(h->w / dim_block.x, h->h / dim_block.y);

	// Invoke kernel
	heatmap_render_saturated<<<dim_grid, dim_block>>>(&d_colorscheme, saturation, h->h, d_bufarray, d_colorbuf);

	cudaMemcpy(colorbuf, d_colorbuf, d_colorbuf_size, cudaMemcpyDeviceToHost);
}