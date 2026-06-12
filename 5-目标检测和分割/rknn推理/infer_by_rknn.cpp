#include "../../include/RKNN_InferenceEngine/infer_by_rknn.h"
#include <iostream>


InferByRKNN::InferByRKNN(std::string model_path)
{
  int ret =0;
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
  ret = init_rknn_model(model_path.c_str(),&rknn_app_ctx);
  if (ret != 0) {
    printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path.c_str());
    release_rknn_model(&rknn_app_ctx);
  }

}

InferByRKNN::~InferByRKNN()
{
  int ret =0;
  ret = release_rknn_model(&rknn_app_ctx);
  if (ret !=0)
  {
    printf("release_rknn_model fail! ret=%d\n", ret);
    return ;
  }
  printf("release_rknn_model successfully! ret=%d\n", ret);
}

int InferByRKNN::init_rknn_model(std::string model_path,rknn_app_context_t *app_ctx)
{
  int ret;
  int model_len = 0;
  char* model;
  rknn_context ctx=0;

  // 加载模型
  model_len = read_data_from_file(model_path.c_str(),&model);
  if (model == NULL)
  {
    printf("load model fail!\n");
    return -1;
  }

  // 初始化模型
  ret = rknn_init(&ctx,model,model_len,0,NULL);
  free(model);
  if (ret <0)
  {
    printf("rknn init fail!ret=%d\n", ret);
    return -1;
  }

  //获取输入输出数量
  rknn_input_output_num io_num;
  ret = rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io_num,sizeof(io_num));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail!ret=%d\n", ret);
    return -1;
  }

  printf("model input num: %d, output num: %d\n", io_num.n_input,io_num.n_output);

  // 获取输入信息
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs,0,sizeof(input_attrs));
  for (int i=0; i<io_num.n_input;i++)
  {
    input_attrs[i].index =i;
    ret = rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&(input_attrs[i]),sizeof(rknn_tensor_attr));
    if (ret!=RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n",ret);
      return -1;
    }
  }

  // 获取输出信息
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs,0,sizeof(output_attrs));
  for (int i=0; i< io_num.n_output;i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&(output_attrs[i]),sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n",ret);
      return -1;
    }
  }

  // set to context
  app_ctx->rknn_ctx = ctx;
  if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8)
  {
    app_ctx->is_quant = true;
  } else {
    app_ctx->is_quant = false;
  }


  app_ctx->io_num = io_num;
  app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input*sizeof(rknn_tensor_attr));
  memcpy(app_ctx->input_attrs,input_attrs,io_num.n_input*sizeof(rknn_tensor_attr));
  app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output*sizeof(rknn_tensor_attr));
  memcpy(app_ctx->output_attrs,output_attrs,io_num.n_output*sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    app_ctx->model_channel = input_attrs[0].dims[1];
    app_ctx->model_height = input_attrs[0].dims[2];
    app_ctx->model_width = input_attrs[0].dims[3];
  } else {
    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];
  }

  printf("model input height=%d,width=%d,channel=%d\n",app_ctx->model_height,app_ctx->model_width,app_ctx->model_channel);
  return 0;
}

int InferByRKNN::read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("fopen %s fail!\n", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *)malloc(file_size + 1);
  data[file_size] = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp)) {
    printf("fread %s fail!\n", path);
    free(data);
    fclose(fp);
    return -1;
  }
  if (fp) {
    fclose(fp);
  }
  *out_data = data;
  return file_size;
}

int InferByRKNN::release_rknn_model(rknn_app_context_t *app_ctx) {
  if (app_ctx->input_attrs != NULL) {
    free(app_ctx->input_attrs);
    app_ctx->input_attrs = NULL;
  }
  if (app_ctx->output_attrs != NULL) {
    free(app_ctx->output_attrs);
    app_ctx->output_attrs = NULL;
  }
  if (app_ctx->rknn_ctx != 0) {
    rknn_destroy(app_ctx->rknn_ctx);
    app_ctx->rknn_ctx = 0;
  }
  return 0;
}


bool InferByRKNN::infer(const cv::Mat &image,std::vector<RknnOutputTensor> &outputs)
{
  int ret = 0;
  // image 转为 tensor
  int input_height_ = image.rows;
  int input_width_ = image.cols;
  int input_channels_ = image.channels();

  std::vector<float> tensor;
  tensor.resize(static_cast<size_t>(input_channels_*input_height_*input_width_));

  if (rknn_app_ctx.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
      std::vector<cv::Mat> channels;
      cv::split(image, channels);

      const size_t plane_size = static_cast<size_t>(input_height_ * input_width_);
      for (int channel = 0; channel < input_channels_; ++channel) {
          std::memcpy(tensor.data() + static_cast<size_t>(channel) * plane_size,
                      channels[channel].ptr<float>(), plane_size * sizeof(float));
      }
  } else {
      std::memcpy(tensor.data(), image.ptr<float>(),
                  tensor.size() * sizeof(float));
  }

  // 设置输入和输出

  rknn_input inputs[rknn_app_ctx.io_num.n_input];
  rknn_output rknn_outputs[rknn_app_ctx.io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(rknn_outputs, 0, sizeof(rknn_outputs));

  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_FLOAT32;
  inputs[0].fmt = rknn_app_ctx.input_attrs[0].fmt;
  inputs[0].size = static_cast<uint32_t>(tensor.size() * sizeof(float));
  inputs[0].buf = const_cast<float *>(tensor.data());

  ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return false;
  }


  // 执行推理
  ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return false;
  }


  // 获取输出
  memset(rknn_outputs, 0, sizeof(rknn_outputs));
  for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++) {
      rknn_outputs[i].index = i;
      rknn_outputs[i].want_float = 1;
  }

  ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, rknn_outputs, NULL);
  if (ret != RKNN_SUCC) {
      std::cerr << "rknn_outputs_get failed, ret=" << ret << std::endl;
      return false;
  }

  // 组合结果
  outputs.clear();
  outputs.reserve(rknn_app_ctx.io_num.n_output);
  for (size_t index=0;index< rknn_app_ctx.io_num.n_output;++index)
  {
    const rknn_tensor_attr &attr = rknn_app_ctx.output_attrs[index];
    const float *output_ptr = static_cast<const float *>(rknn_outputs[index].buf);

    RknnOutputTensor tensor;
    tensor.name = attr.name;
    tensor.shape = tensor_shape_from_attr(attr);
    tensor.values.assign(output_ptr,output_ptr +attr.n_elems);
    outputs.push_back(std::move(tensor));
  }

  rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, rknn_outputs);
  return true;
}

std::vector<int> InferByRKNN::tensor_shape_from_attr(const rknn_tensor_attr &attr) {
    std::vector<int> shape;
    shape.reserve(attr.n_dims);
    for (uint32_t index = 0; index < attr.n_dims; ++index) {
        shape.push_back(static_cast<int>(attr.dims[index]));
    }
    return shape;
}

int InferByRKNN::pipeline(const cv::Mat image,std::vector<RknnOutputTensor> &outputs)
{
  cv::Mat pre_image;
  pre_image = preprocess(image);
  if (pre_image.empty()) {
    std::cerr << "preprocess failed." << std::endl;
    return -1;
  }
  if (!infer(pre_image, outputs)) {
    std::cerr << "infer failed." << std::endl;
    return -1;
  }
  return 0;
}


cv::Mat InferByRKNN::preprocess(const cv::Mat &image)
{
    if (image.empty()) {
        std::cerr << "Input image is empty." << std::endl;
        return cv::Mat();
    }

    if (image.channels() != 3) {
        std::cerr << "Only 3-channel BGR images are supported." << std::endl;
        return cv::Mat();
    }

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const int source_h = rgb.rows;
    const int source_w = rgb.cols;
    int resized_w = rknn_app_ctx.model_width;
    int resized_h = rknn_app_ctx.model_height;

    if (source_h < source_w) {
        resized_h = rknn_app_ctx.model_height;
        resized_w = static_cast<int>(std::round(
            static_cast<double>(source_w) *
            static_cast<double>(rknn_app_ctx.model_height) /
            static_cast<double>(source_h)));
    } else {
        resized_w = rknn_app_ctx.model_width;
        resized_h = static_cast<int>(std::round(
            static_cast<double>(source_h) *
            static_cast<double>(rknn_app_ctx.model_width) /
            static_cast<double>(source_w)));
    }

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    const int top = std::max(0, (resized_h - rknn_app_ctx.model_height) / 2);
    const int left = std::max(0, (resized_w - rknn_app_ctx.model_width) / 2);
    cv::Rect roi(left, top, rknn_app_ctx.model_width, rknn_app_ctx.model_height);

    if (roi.x + roi.width > resized.cols || roi.y + roi.height > resized.rows) {
        std::cerr << "Invalid crop ROI during preprocessing." << std::endl;
        return cv::Mat();
    }

    cv::Mat cropped = resized(roi).clone();

    cv::Mat float_image;
    cropped.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    return float_image;
}