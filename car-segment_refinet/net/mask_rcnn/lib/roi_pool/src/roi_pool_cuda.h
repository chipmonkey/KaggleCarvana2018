int roi_pool_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * argmax);

int roi_pool_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * argmax);