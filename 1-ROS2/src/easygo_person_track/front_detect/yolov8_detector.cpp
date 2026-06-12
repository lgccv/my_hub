#include "yolov8_detector.h"
#include "common.h"

Yolov8Detector::Yolov8Detector() {
  memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
}

Yolov8Detector::Yolov8Detector(std::string model_path) : Yolov8Detector() {
  int ret;
  ret = init_yolov8_model(model_path.c_str(), &rknn_app_ctx);
  if (ret != 0) {
    printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret,
           model_path.c_str());
    destory();
  }
}

Yolov8Detector::~Yolov8Detector() { destory(); }

bool Yolov8Detector::destory() {
  int ret;
  ret = release_yolov8_model(&rknn_app_ctx);
  if (ret != 0) {
    printf("release_yolov8_model fail! ret=%d\n", ret);
    return ret;
  }

  return 0;
}

bool Yolov8Detector::detect(const cv::Mat &image,
                            object_detect_result_list &objects,
                            const float prob_threshold,
                            const float nms_threshold) {
  int ret;
  if (image.empty()) {
    printf("detect fail! input image is empty.\n");
    return false;
  }

  cv::Mat local_copy = image.clone();
  cv::Mat input_image;
  letterbox_t letter_box_p;

  if (image.channels() == 3) {
    // 先做letterbox,再做BGR2RGB
    ret = letter_box_by_opencv(
        local_copy, input_image,
        cv::Size(rknn_app_ctx.model_width, rknn_app_ctx.model_height),
        &letter_box_p, cv::Scalar(114, 114, 114));
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
  } else {
    printf("detect fail! unsupported channel count=%d\n", image.channels());
    return false;
  }

  if (!input_image.isContinuous()) {
    input_image = input_image.clone();
  }
  const int image_size =
      static_cast<int>(input_image.total() * input_image.elemSize());

  image_buffer_t src_image;
  memset(&src_image, 0, sizeof(src_image));
  src_image.virt_addr = input_image.data;
  src_image.width = input_image.cols;
  src_image.height = input_image.rows;
  src_image.format = IMAGE_FORMAT_RGB888;
  src_image.size = image_size;

  ret = inference_yolov8_model(&rknn_app_ctx, &src_image, letter_box_p,
                               &objects, prob_threshold, nms_threshold);

  if (ret != 0) {
    printf("inference_yolov8_model fail! ret=%d\n", ret);
    return false;
  }
  return true;
}

int Yolov8Detector::init_yolov8_model(std::string model_path,
                                      rknn_app_context_t *app_ctx) {
  int ret;
  int model_len = 0;
  char *model;
  rknn_context ctx = 0;

  // 加载模型
  model_len = read_data_from_file(model_path.c_str(), &model);
  if (model == NULL) {
    printf("load model fail!\n");
    return -1;
  }

  // 初始化模型
  ret = rknn_init(&ctx, model, model_len, 0, NULL);
  free(model);
  if (ret < 0) {
    printf("rknn init fail!ret=%d\n", ret);
    return -1;
  }

  // 获取输入输出数量
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail!ret=%d\n", ret);
    return -1;
  }

  printf("model input num: %d, output num: %d\n", io_num.n_input,
         io_num.n_output);

  // 获取输入信息
  // printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail!ret=%d\n", ret);
      return -1;
    }
    // dump_tensor_attr(&(input_attrs[i]));
  }

  // Get Model Output Info
  // printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    // dump_tensor_attr(&(output_attrs[i]));
  }

  // Set to context
  app_ctx->rknn_ctx = ctx;

  // TODO
  if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
      output_attrs[0].type == RKNN_TENSOR_INT8) {
    app_ctx->is_quant = true;
    // app_ctx->is_quant = false;
  } else {
    app_ctx->is_quant = false;
  }

  app_ctx->io_num = io_num;
  app_ctx->input_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->input_attrs, input_attrs,
         io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx->output_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->output_attrs, output_attrs,
         io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    // printf("model is NCHW input fmt\n");
    app_ctx->model_channel = input_attrs[0].dims[1];
    app_ctx->model_height = input_attrs[0].dims[2];
    app_ctx->model_width = input_attrs[0].dims[3];
  } else {
    // printf("model is NHWC input fmt\n");
    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", app_ctx->model_height,
         app_ctx->model_width, app_ctx->model_channel);

  return 0;
}

int Yolov8Detector::release_yolov8_model(rknn_app_context_t *app_ctx) {
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

int Yolov8Detector::inference_yolov8_model(
    rknn_app_context_t *app_ctx, image_buffer_t *img, letterbox_t &letter_box,
    object_detect_result_list *od_results, float prob_thresold_,
    float nms_threshold_) {
  int ret;
  rknn_input inputs[app_ctx->io_num.n_input];
  rknn_output outputs[app_ctx->io_num.n_output];

  const float nms_threshold = nms_threshold_;
  const float box_conf_threshold = prob_thresold_;

  if ((!app_ctx) || (!img) || (!od_results)) {
    return -1;
  }

  memset(od_results, 0x00, sizeof(*od_results));
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));

  // Set Input Data
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size =
      app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
  inputs[0].buf = img->virt_addr;

  ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  // Run
  // printf("rknn_run\n");
  ret = rknn_run(app_ctx->rknn_ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  // Get Output
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < app_ctx->io_num.n_output; i++) {
    outputs[i].index = i;
    outputs[i].want_float = (!app_ctx->is_quant);
    // outputs[i].want_float = true;
  }
  ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs,
                         NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    goto out;
  }

  // Post Process
  post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold,
               od_results);

  // Remeber to release rknn output
  rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

out:

  return 0;
}

void Yolov8Detector::dump_tensor_attr(rknn_tensor_attr *attr) {
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, "
         "size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
         attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
         get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int Yolov8Detector::read_data_from_file(const char *path, char **out_data) {
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

int Yolov8Detector::get_image_size(image_buffer_t *image) {
  if (image == NULL) {
    return 0;
  }
  switch (image->format) {
  case IMAGE_FORMAT_GRAY8:
    return image->width * image->height;
  case IMAGE_FORMAT_RGB888:
    return image->width * image->height * 3;
  case IMAGE_FORMAT_RGBA8888:
    return image->width * image->height * 4;
  case IMAGE_FORMAT_YUV420SP_NV12:
  case IMAGE_FORMAT_YUV420SP_NV21:
    return image->width * image->height * 3 / 2;
  default:
    break;
  }
  return 0;
}

int Yolov8Detector::get_rga_fmt(image_format_t fmt) {
  switch (fmt) {
  case IMAGE_FORMAT_RGB888:
    return RK_FORMAT_RGB_888;
  case IMAGE_FORMAT_RGBA8888:
    return RK_FORMAT_RGBA_8888;
  case IMAGE_FORMAT_YUV420SP_NV12:
    return RK_FORMAT_YCbCr_420_SP;
  case IMAGE_FORMAT_YUV420SP_NV21:
    return RK_FORMAT_YCrCb_420_SP;
  default:
    return -1;
  }
}

int Yolov8Detector::convert_image_rga(image_buffer_t *src_img,
                                      image_buffer_t *dst_img,
                                      image_rect_t *src_box,
                                      image_rect_t *dst_box, char color) {
  int ret = 0;

  int srcWidth = src_img->width;
  int srcHeight = src_img->height;
  void *src = src_img->virt_addr;
  int src_fd = src_img->fd;
  void *src_phy = NULL;
  int srcFmt = get_rga_fmt(src_img->format);

  int dstWidth = dst_img->width;
  int dstHeight = dst_img->height;
  void *dst = dst_img->virt_addr;
  int dst_fd = dst_img->fd;
  void *dst_phy = NULL;
  int dstFmt = get_rga_fmt(dst_img->format);

  int rotate = 0;

  int use_handle = 0;

  // printf("src width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
  //     srcWidth, srcHeight, srcFmt, src, src_fd);
  // printf("dst width=%d height=%d fmt=0x%x virAddr=0x%p fd=%d\n",
  //     dstWidth, dstHeight, dstFmt, dst, dst_fd);
  // printf("rotate=%d\n", rotate);

  int usage = 0;
  IM_STATUS ret_rga = IM_STATUS_NOERROR;

  // set rga usage
  usage |= rotate;

  // set rga rect
  im_rect srect;
  im_rect drect;
  im_rect prect;
  memset(&prect, 0, sizeof(im_rect));

  if (src_box != NULL) {
    srect.x = src_box->left;
    srect.y = src_box->top;
    srect.width = src_box->right - src_box->left + 1;
    srect.height = src_box->bottom - src_box->top + 1;
  } else {
    srect.x = 0;
    srect.y = 0;
    srect.width = srcWidth;
    srect.height = srcHeight;
  }

  if (dst_box != NULL) {
    drect.x = dst_box->left;
    drect.y = dst_box->top;
    drect.width = dst_box->right - dst_box->left + 1;
    drect.height = dst_box->bottom - dst_box->top + 1;
  } else {
    drect.x = 0;
    drect.y = 0;
    drect.width = dstWidth;
    drect.height = dstHeight;
  }

  // set rga buffer
  rga_buffer_t rga_buf_src;
  rga_buffer_t rga_buf_dst;
  rga_buffer_t pat;
  rga_buffer_handle_t rga_handle_src = 0;
  rga_buffer_handle_t rga_handle_dst = 0;
  memset(&pat, 0, sizeof(rga_buffer_t));

  im_handle_param_t in_param;
  in_param.width = srcWidth;
  in_param.height = srcHeight;
  in_param.format = srcFmt;

  im_handle_param_t dst_param;
  dst_param.width = dstWidth;
  dst_param.height = dstHeight;
  dst_param.format = dstFmt;

  if (use_handle) {
    if (src_phy != NULL) {
      rga_handle_src = importbuffer_physicaladdr((uint64_t)src_phy, &in_param);
    } else if (src_fd > 0) {
      rga_handle_src = importbuffer_fd(src_fd, &in_param);
    } else {
      rga_handle_src = importbuffer_virtualaddr(src, &in_param);
    }
    if (rga_handle_src <= 0) {
      printf("src handle error %d\n", rga_handle_src);
      ret = -1;
      goto err;
    }
    rga_buf_src = wrapbuffer_handle(rga_handle_src, srcWidth, srcHeight, srcFmt,
                                    srcWidth, srcHeight);
  } else {
    if (src_phy != NULL) {
      rga_buf_src = wrapbuffer_physicaladdr(src_phy, srcWidth, srcHeight,
                                            srcFmt, srcWidth, srcHeight);
    } else if (src_fd > 0) {
      rga_buf_src = wrapbuffer_fd(src_fd, srcWidth, srcHeight, srcFmt, srcWidth,
                                  srcHeight);
    } else {
      rga_buf_src = wrapbuffer_virtualaddr(src, srcWidth, srcHeight, srcFmt,
                                           srcWidth, srcHeight);
    }
  }

  if (use_handle) {
    if (dst_phy != NULL) {
      rga_handle_dst = importbuffer_physicaladdr((uint64_t)dst_phy, &dst_param);
    } else if (dst_fd > 0) {
      rga_handle_dst = importbuffer_fd(dst_fd, &dst_param);
    } else {
      rga_handle_dst = importbuffer_virtualaddr(dst, &dst_param);
    }
    if (rga_handle_dst <= 0) {
      printf("dst handle error %d\n", rga_handle_dst);
      ret = -1;
      goto err;
    }
    rga_buf_dst = wrapbuffer_handle(rga_handle_dst, dstWidth, dstHeight, dstFmt,
                                    dstWidth, dstHeight);
  } else {
    if (dst_phy != NULL) {
      rga_buf_dst = wrapbuffer_physicaladdr(dst_phy, dstWidth, dstHeight,
                                            dstFmt, dstWidth, dstHeight);
    } else if (dst_fd > 0) {
      rga_buf_dst = wrapbuffer_fd(dst_fd, dstWidth, dstHeight, dstFmt, dstWidth,
                                  dstHeight);
    } else {
      rga_buf_dst = wrapbuffer_virtualaddr(dst, dstWidth, dstHeight, dstFmt,
                                           dstWidth, dstHeight);
    }
  }

  if (drect.width != dstWidth || drect.height != dstHeight) {
    im_rect dst_whole_rect = {0, 0, dstWidth, dstHeight};
    int imcolor = 0;
    char *p_imcolor = reinterpret_cast<char *>(&imcolor);
    p_imcolor[0] = reinterpret_cast<char>(color);
    p_imcolor[1] = reinterpret_cast<char>(color);
    p_imcolor[2] = reinterpret_cast<char>(color);
    p_imcolor[3] = reinterpret_cast<char>(color);
    // printf("fill dst image (x y w h)=(%d %d %d %d) with color=0x%x\n",
    // dst_whole_rect.x, dst_whole_rect.y, dst_whole_rect.width,
    // dst_whole_rect.height, imcolor);
    // ret_rga = imfill(rga_buf_dst, dst_whole_rect, imcolor);
    ret_rga = static_cast<IM_STATUS>(-1);
    if (ret_rga <= 0) {
      if (dst != NULL) {
        size_t dst_size = get_image_size(dst_img);
        memset(dst, color, dst_size);
      } else {
        printf("Warning: Can not fill color on target image\n");
      }
    }
  }

  // rga process
  ret_rga =
      improcess(rga_buf_src, rga_buf_dst, pat, srect, drect, prect, usage);
  if (ret_rga <= 0) {
    printf("Error on improcess STATUS=%d\n", ret_rga);
    printf("RGA error message: %s\n", imStrError((IM_STATUS)ret_rga));
    ret = -1;
  }

err:
  if (rga_handle_src > 0) {
    releasebuffer_handle(rga_handle_src);
  }

  if (rga_handle_dst > 0) {
    releasebuffer_handle(rga_handle_dst);
  }

  // printf("finish\n");
  return ret;
}

int Yolov8Detector::crop_and_scale_image_c(
    int channel, unsigned char *src, int src_width, int src_height, int crop_x,
    int crop_y, int crop_width, int crop_height, unsigned char *dst,
    int dst_width, int dst_height, int dst_box_x, int dst_box_y,
    int dst_box_width, int dst_box_height) {
  if (dst == NULL) {
    printf("dst buffer is null\n");
    return -1;
  }

  float x_ratio = (float)crop_width / (float)dst_box_width;
  float y_ratio = (float)crop_height / (float)dst_box_height;

  // printf("src_width=%d src_height=%d crop_x=%d crop_y=%d crop_width=%d
  // crop_height=%d\n",
  //     src_width, src_height, crop_x, crop_y, crop_width, crop_height);
  // printf("dst_width=%d dst_height=%d dst_box_x=%d dst_box_y=%d
  // dst_box_width=%d dst_box_height=%d\n",
  //     dst_width, dst_height, dst_box_x, dst_box_y, dst_box_width,
  //     dst_box_height);
  // printf("channel=%d x_ratio=%f y_ratio=%f\n", channel, x_ratio, y_ratio);

  // 从原图指定区域取数据，双线性缩放到目标指定区域
  for (int dst_y = dst_box_y; dst_y < dst_box_y + dst_box_height; dst_y++) {
    for (int dst_x = dst_box_x; dst_x < dst_box_x + dst_box_width; dst_x++) {
      int dst_x_offset = dst_x - dst_box_x;
      int dst_y_offset = dst_y - dst_box_y;

      int src_x = (int)(dst_x_offset * x_ratio) + crop_x;
      int src_y = (int)(dst_y_offset * y_ratio) + crop_y;

      float x_diff = (dst_x_offset * x_ratio) - (src_x - crop_x);
      float y_diff = (dst_y_offset * y_ratio) - (src_y - crop_y);

      int index1 = src_y * src_width * channel + src_x * channel;
      int index2 = index1 + src_width * channel; // down
      if (src_y == src_height - 1) {
        // 如果到图像最下边缘，变成选择上面的像素
        index2 = index1 - src_width * channel;
      }
      int index3 = index1 + 1 * channel; // right
      int index4 = index2 + 1 * channel; // down right
      if (src_x == src_width - 1) {
        // 如果到图像最右边缘，变成选择左边的像素
        index3 = index1 - 1 * channel;
        index4 = index2 - 1 * channel;
      }

      // printf("dst_x=%d dst_y=%d dst_x_offset=%d dst_y_offset=%d src_x=%d
      // src_y=%d x_diff=%f y_diff=%f src index=%d %d %d %d\n",
      //     dst_x, dst_y, dst_x_offset, dst_y_offset,
      //     src_x, src_y, x_diff, y_diff,
      //     index1, index2, index3, index4);

      for (int c = 0; c < channel; c++) {
        unsigned char A = src[index1 + c];
        unsigned char B = src[index3 + c];
        unsigned char C = src[index2 + c];
        unsigned char D = src[index4 + c];

        unsigned char pixel =
            (unsigned char)(A * (1 - x_diff) * (1 - y_diff) +
                            B * x_diff * (1 - y_diff) +
                            C * y_diff * (1 - x_diff) + D * x_diff * y_diff);

        dst[(dst_y * dst_width + dst_x) * channel + c] = pixel;
      }
    }
  }

  return 0;
}

int Yolov8Detector::crop_and_scale_image_yuv420sp(
    unsigned char *src, int src_width, int src_height, int crop_x, int crop_y,
    int crop_width, int crop_height, unsigned char *dst, int dst_width,
    int dst_height, int dst_box_x, int dst_box_y, int dst_box_width,
    int dst_box_height) {

  unsigned char *src_y = src;
  unsigned char *src_uv = src + src_width * src_height;

  unsigned char *dst_y = dst;
  unsigned char *dst_uv = dst + dst_width * dst_height;

  crop_and_scale_image_c(1, src_y, src_width, src_height, crop_x, crop_y,
                         crop_width, crop_height, dst_y, dst_width, dst_height,
                         dst_box_x, dst_box_y, dst_box_width, dst_box_height);

  crop_and_scale_image_c(2, src_uv, src_width / 2, src_height / 2, crop_x / 2,
                         crop_y / 2, crop_width / 2, crop_height / 2, dst_uv,
                         dst_width / 2, dst_height / 2, dst_box_x, dst_box_y,
                         dst_box_width, dst_box_height);

  return 0;
}

int Yolov8Detector::convert_image_cpu(image_buffer_t *src, image_buffer_t *dst,
                                      image_rect_t *src_box,
                                      image_rect_t *dst_box, char color) {
  int ret;
  if (dst->virt_addr == NULL) {
    return -1;
  }
  if (src->virt_addr == NULL) {
    return -1;
  }
  if (src->format != dst->format) {
    return -1;
  }

  int src_box_x = 0;
  int src_box_y = 0;
  int src_box_w = src->width;
  int src_box_h = src->height;
  if (src_box != NULL) {
    src_box_x = src_box->left;
    src_box_y = src_box->top;
    src_box_w = src_box->right - src_box->left + 1;
    src_box_h = src_box->bottom - src_box->top + 1;
  }
  int dst_box_x = 0;
  int dst_box_y = 0;
  int dst_box_w = dst->width;
  int dst_box_h = dst->height;
  if (dst_box != NULL) {
    dst_box_x = dst_box->left;
    dst_box_y = dst_box->top;
    dst_box_w = dst_box->right - dst_box->left + 1;
    dst_box_h = dst_box->bottom - dst_box->top + 1;
  }

  // fill pad color
  if (dst_box_w != dst->width || dst_box_h != dst->height) {
    int dst_size = get_image_size(dst);
    memset(dst->virt_addr, color, dst_size);
  }

  int need_release_dst_buffer = 0;
  int reti = 0;
  if (src->format == IMAGE_FORMAT_RGB888) {
    reti = crop_and_scale_image_c(3, src->virt_addr, src->width, src->height,
                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                  dst->virt_addr, dst->width, dst->height,
                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
  } else if (src->format == IMAGE_FORMAT_RGBA8888) {
    reti = crop_and_scale_image_c(4, src->virt_addr, src->width, src->height,
                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                  dst->virt_addr, dst->width, dst->height,
                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
  } else if (src->format == IMAGE_FORMAT_GRAY8) {
    reti = crop_and_scale_image_c(1, src->virt_addr, src->width, src->height,
                                  src_box_x, src_box_y, src_box_w, src_box_h,
                                  dst->virt_addr, dst->width, dst->height,
                                  dst_box_x, dst_box_y, dst_box_w, dst_box_h);
  } else if (src->format == IMAGE_FORMAT_YUV420SP_NV12 ||
             src->format == IMAGE_FORMAT_YUV420SP_NV21) {
    reti = crop_and_scale_image_yuv420sp(
        src->virt_addr, src->width, src->height, src_box_x, src_box_y,
        src_box_w, src_box_h, dst->virt_addr, dst->width, dst->height,
        dst_box_x, dst_box_y, dst_box_w, dst_box_h);
  } else {
    printf("no support format %d\n", src->format);
  }
  if (reti != 0) {
    printf("convert_image_cpu fail %d\n", reti);
    return -1;
  }
  printf("finish\n");
  return 0;
}

int Yolov8Detector::convert_image(image_buffer_t *src_img,
                                  image_buffer_t *dst_img,
                                  image_rect_t *src_box, image_rect_t *dst_box,
                                  char color) {
  int ret;
  if (src_img->width % 16 == 0 && dst_img->width % 16 == 0) {
    ret = convert_image_rga(src_img, dst_img, src_box, dst_box, color);
    if (ret != 0) {
      printf("try convert image use cpu\n");
      ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
    }
  } else {
    printf("src width is not 4/16-aligned, convert image use cpu\n");
    ret = convert_image_cpu(src_img, dst_img, src_box, dst_box, color);
  }
  return ret;
}

int32_t Yolov8Detector::__clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

int8_t Yolov8Detector::qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

void Yolov8Detector::compute_dfl(float *tensor, int dfl_len, float *box) {
  for (int b = 0; b < 4; b++) {
    float exp_t[dfl_len];
    float exp_sum = 0;
    float acc_sum = 0;
    for (int i = 0; i < dfl_len; i++) {
      exp_t[i] = exp(tensor[i + b * dfl_len]);
      exp_sum += exp_t[i];
    }

    for (int i = 0; i < dfl_len; i++) {
      acc_sum += exp_t[i] / exp_sum * i;
    }
    box[b] = acc_sum;
  }
}

float Yolov8Detector::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

int Yolov8Detector::process_i8(
    int8_t *box_tensor, int32_t box_zp, float box_scale, int8_t *score_tensor,
    int32_t score_zp, float score_scale, int8_t *score_sum_tensor,
    int32_t score_sum_zp, float score_sum_scale, int grid_h, int grid_w,
    int stride, int dfl_len, std::vector<float> &boxes,
    std::vector<float> &objProbs, std::vector<int> &classId, float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
  int8_t score_sum_thres_i8 =
      qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      (void)score_sum_tensor;
      (void)score_sum_thres_i8;

      // 通过 score sum 起到快速过滤的作用
      // if (score_sum_tensor != nullptr) {
      //   if (score_sum_tensor[offset] < score_sum_thres_i8) {
      //     continue;
      //   }
      // }

      int8_t max_score = -score_zp;
      for (int c = 0; c < obj_class_num; c++) {
        if ((score_tensor[offset] > score_thres_i8) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      // compute box
      if (max_score > score_thres_i8) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {
          before_dfl[k] =
              deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);

        objProbs.push_back(
            deqnt_affine_to_f32(max_score, score_zp, score_scale));
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

int Yolov8Detector::process_fp32(float *box_tensor, float *score_tensor,
                                 float *score_sum_tensor, int grid_h,
                                 int grid_w, int stride, int dfl_len,
                                 std::vector<float> &boxes,
                                 std::vector<float> &objProbs,
                                 std::vector<int> &classId, float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;
      int max_class_id = -1;

      (void)score_sum_tensor;
      // 通过 score sum 起到快速过滤的作用
      // if (score_sum_tensor != nullptr) {
      //   if (score_sum_tensor[offset] < threshold) {
      //     continue;
      //   }
      // }

      float max_score = 0;
      for (int c = 0; c < obj_class_num; c++) {
        if ((score_tensor[offset] > threshold) &&
            (score_tensor[offset] > max_score)) {
          max_score = score_tensor[offset];
          max_class_id = c;
        }
        offset += grid_len;
      }

      // compute box
      if (max_score > threshold) {
        offset = i * grid_w + j;
        float box[4];
        float before_dfl[dfl_len * 4];
        for (int k = 0; k < dfl_len * 4; k++) {
          before_dfl[k] = box_tensor[offset];
          offset += grid_len;
        }
        compute_dfl(before_dfl, dfl_len, box);

        float x1, y1, x2, y2, w, h;
        x1 = (-box[0] + j + 0.5) * stride;
        y1 = (-box[1] + i + 0.5) * stride;
        x2 = (box[2] + j + 0.5) * stride;
        y2 = (box[3] + i + 0.5) * stride;
        w = x2 - x1;
        h = y2 - y1;
        boxes.push_back(x1);
        boxes.push_back(y1);
        boxes.push_back(w);
        boxes.push_back(h);

        objProbs.push_back(max_score);
        classId.push_back(max_class_id);
        validCount++;
      }
    }
  }
  return validCount;
}

int Yolov8Detector::quick_sort_indice_inverse(std::vector<float> &input,
                                              int left, int right,
                                              std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

// float Yolov8Detector::CalculateOverlap(float xmin0, float ymin0, float xmax0,
//                                        float ymax0, float xmin1, float ymin1,
//                                        float xmax1, float ymax1) {
//   float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
//   float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
//   float i = w * h;
//   float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
//             (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
//   return u <= 0.f ? 0.f : (i / u);
// }

float Yolov8Detector::CalculateOverlap(float xmin0, float ymin0, float xmax0,
                                       float ymax0, float xmin1, float ymin1,
                                       float xmax1, float ymax1) {
  float w0 = xmax0 - xmin0;
  float h0 = ymax0 - ymin0;
  float w1 = xmax1 - xmin1;
  float h1 = ymax1 - ymin1;
  float inter_w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 0.00001f);
  float inter_h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 0.00001f);
  float inter = inter_w * inter_h;
  float area0 = w0 * h0;
  float area1 = w1 * h1;
  float uni = area0 + area1 - inter;
  return uni <= 0.f ? 0.f : (inter / uni);
}

int Yolov8Detector::nms(int validCount, std::vector<float> &outputLocations,
                        std::vector<int> classIds, std::vector<int> &order,
                        int filterId, float threshold) {
  for (int i = 0; i < validCount; ++i) {
    int n = order[i];
    if (n == -1 || classIds[n] != filterId) {
      continue;
    }
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[m] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

int Yolov8Detector::post_process(rknn_app_context_t *app_ctx, void *outputs,
                                 letterbox_t *letter_box, float conf_threshold,
                                 float nms_threshold,
                                 object_detect_result_list *od_results) {

  rknn_output *_outputs = (rknn_output *)outputs;
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int model_in_w = app_ctx->model_width;
  int model_in_h = app_ctx->model_height;

  memset(od_results, 0, sizeof(object_detect_result_list));

  // default 3 branch

  int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
  int output_per_branch = app_ctx->io_num.n_output / 3;
  for (int i = 0; i < 3; i++) {
    void *score_sum = nullptr;
    int32_t score_sum_zp = 0;
    float score_sum_scale = 1.0;
    if (output_per_branch == 3) {
      score_sum = _outputs[i * output_per_branch + 2].buf;
      score_sum_zp = app_ctx->output_attrs[i * output_per_branch + 2].zp;
      score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
    }
    int box_idx = i * output_per_branch;
    int score_idx = i * output_per_branch + 1;

    grid_h = app_ctx->output_attrs[box_idx].dims[2];
    grid_w = app_ctx->output_attrs[box_idx].dims[3];

    stride = model_in_h / grid_h;

    if (app_ctx->is_quant) {

      validCount += process_i8(
          (int8_t *)_outputs[box_idx].buf, app_ctx->output_attrs[box_idx].zp,
          app_ctx->output_attrs[box_idx].scale,
          (int8_t *)_outputs[score_idx].buf,
          app_ctx->output_attrs[score_idx].zp,
          app_ctx->output_attrs[score_idx].scale, (int8_t *)score_sum,
          score_sum_zp, score_sum_scale, grid_h, grid_w, stride, dfl_len,
          filterBoxes, objProbs, classId, conf_threshold);
    } else {
      validCount += process_fp32(
          (float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf,
          (float *)score_sum, grid_h, grid_w, stride, dfl_len, filterBoxes,
          objProbs, classId, conf_threshold);
    }
  }

  // no object detect
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  od_results->count = 0;

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    od_results->results[last_count].box.left =
        (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.top =
        (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].box.right =
        (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.bottom =
        (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;
  return 0;
}

int Yolov8Detector::convert_image_with_letterbox(image_buffer_t *src_image,
                                                 image_buffer_t *dst_image,
                                                 letterbox_t *letterbox,
                                                 char color) {
  int ret = 0;
  int allow_slight_change = 1;
  int src_w = src_image->width;
  int src_h = src_image->height;
  int dst_w = dst_image->width;
  int dst_h = dst_image->height;
  int resize_w = dst_w;
  int resize_h = dst_h;

  int padding_w = 0;
  int padding_h = 0;

  int _left_offset = 0;
  int _top_offset = 0;
  float scale = 1.0;

  image_rect_t src_box;
  src_box.left = 0;
  src_box.top = 0;
  src_box.right = src_image->width - 1;
  src_box.bottom = src_image->height - 1;

  image_rect_t dst_box;
  dst_box.left = 0;
  dst_box.top = 0;
  dst_box.right = dst_image->width - 1;
  dst_box.bottom = dst_image->height - 1;

  float _scale_w = (float)dst_w / src_w;
  float _scale_h = (float)dst_h / src_h;
  if (_scale_w < _scale_h) {
    scale = _scale_w;
    resize_h = (int)src_h * scale;
  } else {
    scale = _scale_h;
    resize_w = (int)src_w * scale;
  }
  // slight change image size for align
  if (allow_slight_change == 1 && (resize_w % 4 != 0)) {
    resize_w -= resize_w % 4;
  }
  if (allow_slight_change == 1 && (resize_h % 2 != 0)) {
    resize_h -= resize_h % 2;
  }
  // padding
  padding_h = dst_h - resize_h;
  padding_w = dst_w - resize_w;
  // center
  if (_scale_w < _scale_h) {
    dst_box.top = padding_h / 2;
    if (dst_box.top % 2 != 0) {
      dst_box.top -= dst_box.top % 2;
      if (dst_box.top < 0) {
        dst_box.top = 0;
      }
    }
    dst_box.bottom = dst_box.top + resize_h - 1;
    _top_offset = dst_box.top;
  } else {
    dst_box.left = padding_w / 2;
    if (dst_box.left % 2 != 0) {
      dst_box.left -= dst_box.left % 2;
      if (dst_box.left < 0) {
        dst_box.left = 0;
      }
    }
    dst_box.right = dst_box.left + resize_w - 1;
    _left_offset = dst_box.left;
  }
  // printf("scale=%f dst_box=(%d %d %d %d) allow_slight_change=%d
  // _left_offset=%d _top_offset=%d padding_w=%d padding_h=%d\n",
  //     scale, dst_box.left, dst_box.top, dst_box.right, dst_box.bottom,
  //     allow_slight_change, _left_offset, _top_offset, padding_w, padding_h);

  // set offset and scale
  if (letterbox != NULL) {
    letterbox->scale = scale;
    letterbox->x_pad = _left_offset;
    letterbox->y_pad = _top_offset;
  }
  // alloc memory buffer for dst image,
  // remember to free
  if (dst_image->virt_addr == NULL && dst_image->fd <= 0) {
    int dst_size = get_image_size(dst_image);
    dst_image->virt_addr = (uint8_t *)malloc(dst_size);
    if (dst_image->virt_addr == NULL) {
      printf("malloc size %d error\n", dst_size);
      return -1;
    }
  }
  ret = convert_image(src_image, dst_image, &src_box, &dst_box, color);
  return ret;
}

cv::Mat Yolov8Detector::draw_objects(const cv::Mat &bgr,
                                     object_detect_result_list &objects) {
  cv::Mat output = bgr.clone();
  if (output.empty()) {
    return output;
  }

  char text[256];
  for (int i = 0; i < objects.count; i++) {
    object_detect_result *det_result = &(objects.results[i]);
    LOG(INFO) << "cls:" << class_name[det_result->cls_id].c_str() << "coord:" << det_result->box.left <<","<<det_result->box.top<<","<<det_result->box.right<<","<<det_result->box.bottom <<"score:"<<det_result->prop;
    
    int x1 = det_result->box.left;
    int y1 = det_result->box.top;
    int x2 = det_result->box.right;
    int y2 = det_result->box.bottom;

    cv::rectangle(output, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(255, 0, 0), 2);

    snprintf(text, sizeof(text), "%s %.2f",
             class_name[det_result->cls_id].c_str(), det_result->prop);
    cv::putText(output, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 0, 255), 2);
  }
  return output;
}

int Yolov8Detector::letter_box_by_opencv(const cv::Mat &src, cv::Mat &dst,
                                         const cv::Size &new_shape,
                                         letterbox_t *letter_box,
                                         const cv::Scalar &pad_color) {
  if (src.empty()) {
    printf("letter_box fail! input image is empty.\n");
    return -1;
  }
  const int src_h = src.rows;
  const int src_w = src.cols;
  const int dst_h = new_shape.height;
  const int dst_w = new_shape.width;
  // Scale ratio
  float r = std::min((float)dst_h / (float)src_h, (float)dst_w / (float)src_w);
  // Compute padding
  int new_unpad_w = (int)std::round(src_w * r);
  int new_unpad_h = (int)std::round(src_h * r);
  float dw = (float)(dst_w - new_unpad_w) / 2.0f;
  float dh = (float)(dst_h - new_unpad_h) / 2.0f;
  cv::Mat resized;
  if (src_w != new_unpad_w || src_h != new_unpad_h) {
    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0,
               cv::INTER_LINEAR);
  } else {
    resized = src.clone();
  }
  int top = (int)std::round(dh - 0.1f);
  int bottom = (int)std::round(dh + 0.1f);
  int left = (int)std::round(dw - 0.1f);
  int right = (int)std::round(dw + 0.1f);
  cv::copyMakeBorder(resized, dst, top, bottom, left, right,
                     cv::BORDER_CONSTANT, pad_color);
  if (letter_box != nullptr) {
    letter_box->scale = r;
    letter_box->x_pad = left;
    letter_box->y_pad = top;
  }
  return 0;
}