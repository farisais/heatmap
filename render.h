#include "heatmap.h"

#ifndef _RENDER_H
#define _RENDER_H

void heatmap_render_saturated_to_gpu(const heatmap_t* h, const heatmap_colorscheme_t* colorscheme, float saturation, unsigned char* colorbuf);

#endif /* _RENDER_H */