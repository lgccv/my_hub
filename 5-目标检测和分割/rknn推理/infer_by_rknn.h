#ifndef INFER_BY_RKNN
#define INFER_BY_RKNN
#include "../third_party/include/rknn_api.h"
#include <cstring>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;


struct RknnOutputTensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> values;
};

class InferByRKNN
{
public:
    InferByRKNN(std::string model_path);
    ~InferByRKNN();

public:
    int init_rknn_model(std::string model_path,rknn_app_context_t *app_ctx);
    bool infer(const cv::Mat &image,std::vector<RknnOutputTensor> &outputs);
    int release_rknn_model(rknn_app_context_t *app_ctx);
    std::vector<int> tensor_shape_from_attr(const rknn_tensor_attr &attr);
    int pipeline(const cv::Mat image,std::vector<RknnOutputTensor> &outputs);
    cv::Mat preprocess(const cv::Mat &image);

private:
    int read_data_from_file(const char *path, char **out_data);

private:
    rknn_app_context_t rknn_app_ctx;
};

#endif