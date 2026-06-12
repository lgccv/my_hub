#ifndef YOLOV8_DETECTOR
#define YOLOV8_DETECTOR

#include <cstddef>
#include "common.h"
#include "im2d.h"
#include "im2d_type.h"
#include "rga.h"
#include "rknn_api.h"
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <string>
#include <set>
#include "common.h"
#include "glog/logging.h"

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

class Yolov8Detector
{
    public:
        Yolov8Detector();

        Yolov8Detector(std::string model_path);

        bool detect(const cv::Mat &image,object_detect_result_list &objects,const float prob_threshold=0.25f,const float nms_threshold = 0.45f);

        cv::Mat draw_objects(const cv::Mat &bgr,object_detect_result_list &objects);

        ~Yolov8Detector();

        bool destory();

    private:
        int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }
        void dump_tensor_attr(rknn_tensor_attr *attr);
        int read_data_from_file(const char *path, char **out_data);
        int get_image_size(image_buffer_t *image);
        int get_rga_fmt(image_format_t fmt);
        int convert_image_rga(image_buffer_t *src_img, image_buffer_t *dst_img,image_rect_t *src_box, image_rect_t *dst_box,char color);
        int crop_and_scale_image_c(int channel, unsigned char *src, int src_width,int src_height, int crop_x, int crop_y,int crop_width, int crop_height, unsigned char *dst,int dst_width, int dst_height, int dst_box_x,int dst_box_y, int dst_box_width,int dst_box_height);
        int crop_and_scale_image_yuv420sp(unsigned char *src, int src_width,int src_height, int crop_x, int crop_y,int crop_width, int crop_height,unsigned char *dst, int dst_width,int dst_height, int dst_box_x, int dst_box_y,int dst_box_width, int dst_box_height);
        int convert_image_cpu(image_buffer_t *src, image_buffer_t *dst,image_rect_t *src_box, image_rect_t *dst_box,char color);
        int convert_image(image_buffer_t *src_img, image_buffer_t *dst_img,image_rect_t *src_box, image_rect_t *dst_box, char color);
        int32_t __clip(float val, float min, float max);
        int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
        void compute_dfl(float *tensor, int dfl_len, float *box);
        float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
        int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,int8_t *score_tensor, int32_t score_zp, float score_scale,int8_t *score_sum_tensor, int32_t score_sum_zp,float score_sum_scale, int grid_h, int grid_w, int stride,int dfl_len, std::vector<float> &boxes,std::vector<float> &objProbs, std::vector<int> &classId,float threshold);
        int process_fp32(float *box_tensor, float *score_tensor,float *score_sum_tensor, int grid_h, int grid_w,int stride, int dfl_len, std::vector<float> &boxes,std::vector<float> &objProbs, std::vector<int> &classId,float threshold);
        int quick_sort_indice_inverse(std::vector<float> &input, int left, int right,std::vector<int> &indices);
        float CalculateOverlap(float xmin0, float ymin0, float xmax0,float ymax0, float xmin1, float ymin1,float xmax1, float ymax1);
        int nms(int validCount, std::vector<float> &outputLocations,std::vector<int> classIds, std::vector<int> &order, int filterId,float threshold);
        int post_process(rknn_app_context_t *app_ctx, void *outputs,letterbox_t *letter_box, float conf_threshold,float nms_threshold, object_detect_result_list *od_results);
        int convert_image_with_letterbox(image_buffer_t *src_image,image_buffer_t *dst_image,letterbox_t *letterbox, char color);
        int init_yolov8_model(std::string model_path, rknn_app_context_t *app_ctx);
        int release_yolov8_model(rknn_app_context_t *app_ctx);
        int inference_yolov8_model(rknn_app_context_t *app_ctx, image_buffer_t *img,letterbox_t& letter_box,object_detect_result_list *od_results,float prob_thresold_, float nms_threshold_);
        int letter_box_by_opencv(const cv::Mat &src, cv::Mat &dst,const cv::Size &new_shape,letterbox_t *letter_box,const cv::Scalar &pad_color);
    private:
        rknn_app_context_t rknn_app_ctx;
        int obj_class_num =2;
        int OBJ_NUMB_MAX_SIZE = 128;
        float bbox_score = 0.25;
        float nms_score = 0.45;
        std::vector<std::string> class_name = {"person","amr"};
};

#endif



