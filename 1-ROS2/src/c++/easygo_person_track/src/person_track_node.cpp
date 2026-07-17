#include "BYTETracker.h"
#include "common.h"
#include "cv_bridge/cv_bridge.hpp"
#include "easygo_follow_msgs/action/start_binding.hpp"
#include "easygo_follow_msgs/msg/perception_event.hpp"
#include "easygo_follow_msgs/msg/tracked_target.hpp"
#include "easygo_follow_msgs/srv/stop_tracking.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "yolov8_detector.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <glog/logging.h>

struct DepthSample {
  double depth_m;
  double u;
  double v;
  int valid_count;

  double ground_x_m;
  double ground_y_m;
  double ground_z_m;
};


struct ValidCandidate {
  const STrack *track;
  cv::Rect box;
  DepthSample sample;
  double real_height_m;
};


class PersonTrackNode : public rclcpp::Node {
public:
  using StartBinding = easygo_follow_msgs::action::StartBinding;
  using GoalHandle = rclcpp_action::ServerGoalHandle<StartBinding>;
  PersonTrackNode() : Node("person_track_node") {
    // 订阅彩色图像话题
    rgb_topic = this->declare_parameter<std::string>("rgb_topic",
                                                     "/camera/color/image_raw");
    subscription_rgb_image = this->create_subscription<sensor_msgs::msg::Image>(
        rgb_topic, 10,
        std::bind(&PersonTrackNode::onReceiveRGB, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s", rgb_topic.c_str());

    // 订阅深度图像话题
    depth_topic = this->declare_parameter<std::string>(
        "depth_topic", "/camera/depth/image_raw");
    subscription_depth_image =
        this->create_subscription<sensor_msgs::msg::Image>(
            depth_topic, 10,
            std::bind(&PersonTrackNode::onReceiveDepth, this,
                      std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s", depth_topic.c_str());

    // 发送结果话题给运控
    tracked_target_topic = this->declare_parameter<std::string>(
        "tracked_target_topic", "/follow/perception/tracked_target");
    pub_to_motion =
        this->create_publisher<easygo_follow_msgs::msg::TrackedTarget>(
            tracked_target_topic, 10);
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s",
                tracked_target_topic.c_str());

    // 发送结果给app
    perception_event_topic = this->declare_parameter<std::string>(
        "perception_event_topic", "/follow/perception/events");
    pub_to_app =
        this->create_publisher<easygo_follow_msgs::msg::PerceptionEvent>(
            perception_event_topic, 10);
    RCLCPP_INFO(this->get_logger(), "Publishing perception events to %s",
                perception_event_topic.c_str());

    // 接收停止跟踪的服务
    stop_track_server_name = this->declare_parameter<std::string>(
        "stop_track_server_name", "/follow/perception/stop_tracking");
    stop_track_server =
        this->create_service<easygo_follow_msgs::srv::StopTracking>(
            stop_track_server_name,
            std::bind(&PersonTrackNode::stop_tracking, this,
                      std::placeholders::_1, std::placeholders::_2));

    // 开始绑定动作
    action_server_name = this->declare_parameter<std::string>(
        "action_server_name", "/follow/perception/start_binding");
    binding_action =
        rclcpp_action::create_server<easygo_follow_msgs::action::StartBinding>(
            this->get_node_base_interface(), this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(), action_server_name,
            std::bind(&PersonTrackNode::handle_goal, this,
                      std::placeholders::_1, std::placeholders::_2),
            std::bind(&PersonTrackNode::handle_cancel, this,
                      std::placeholders::_1),
            std::bind(&PersonTrackNode::handle_accepted, this,
                      std::placeholders::_1));

    // 初始化检测器和跟踪器
    model_path = this->declare_parameter<std::string>(
        "model_path", "./model/last_quant.rknn");
    net = std::make_shared<Yolov8Detector>(model_path);
    tracker = std::make_shared<BYTETracker>();
    reset_track();

    det_conf_threshold =
        this->declare_parameter<double>("det_conf_threshold", 0.25);
    det_nms_threshold =
        this->declare_parameter<double>("det_nms_threshold", 0.45);

    
    // 调试代码
    detect_result_image_topic = this->declare_parameter<std::string>(
      "detect_result_image_topic", "/follow/perception/detection_image");
    pub_detection_image =
      this->create_publisher<sensor_msgs::msg::Image>(
          detect_result_image_topic, 10);

    track_result_image_topic = this->declare_parameter<std::string>(
            "track_result_image_topic", "/follow/perception/tracking_image");
    pub_tracking_image =
            this->create_publisher<sensor_msgs::msg::Image>(
                track_result_image_topic, 10);   

    RCLCPP_INFO(this->get_logger(), "初始化完成");
  }

public:
  void reset_track() {
    std::lock_guard<std::mutex> lock(tracker_mutex);
    tracker->reset(true);
    RCLCPP_INFO(this->get_logger(), "reset_tracker");
  }

  void reset_all_tracking() {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      track_id = -1;
      binding_active = false;
      binding_completed = false;
      binding_stable_frames = 0;
      binding_candidate_track_id = -1;
      tracking_enabled = false;
      stop_requested = false;
      active_goal_handle.reset();
    }
    reset_track();
  }

  // 将检测的输出转为为跟踪的输入
  void decobj_to_trackobj(object_detect_result_list &objects,
                          std::vector<Object> &trackobj) {
    trackobj.clear();
    const int PERSON_CLASS_ID = 0;
    for (int i = 0; i < objects.count; ++i) {
      const auto &obj = objects.results[i];
      if (obj.cls_id != PERSON_CLASS_ID) {
        continue;
      }
      Object trackobj_temp;
      trackobj_temp.classId = obj.cls_id;
      trackobj_temp.score = obj.prop;
      float x = static_cast<float>(obj.box.left);
      float y = static_cast<float>(obj.box.top);
      float w = static_cast<float>(obj.box.right - obj.box.left);
      float h = static_cast<float>(obj.box.bottom - obj.box.top);
      trackobj_temp.box = cv::Rect_<float>(x, y, w, h);
      trackobj.push_back(trackobj_temp);
    }
  }

  cv::Rect track_to_rect(const STrack &track) const {
    if (track.tlwh.size() < 4) {
      return cv::Rect();
    }

    const int x = std::max(0, static_cast<int>(track.tlwh[0]));
    const int y = std::max(0, static_cast<int>(track.tlwh[1]));
    const int w = std::max(0, static_cast<int>(track.tlwh[2]));
    const int h = std::max(0, static_cast<int>(track.tlwh[3]));
    return cv::Rect(x, y, w, h);
  }

  void publish_perception_event(const std_msgs::msg::Header &header,
                                uint8_t event_type, uint16_t code,
                                const std::string &message) {
    if (!pub_to_app) {
      return;
    }
    easygo_follow_msgs::msg::PerceptionEvent event;
    event.header = header;
    event.event_type = event_type;
    event.code = code;
    event.message = message;
    LOG(INFO) << "lgc_publish_perception_event event: "
          << "event_type:" << static_cast<int>(event.event_type)
          << ", code:" << event.code
          << ", message:" << event.message;

    pub_to_app->publish(event);
  }

  bool sample_depth_m(const cv::Mat &depth, const std::string &encoding,
                      const cv::Rect &bbox, DepthSample &sample) const {
    if (depth.empty() || bbox.width <= 0 || bbox.height <= 0) {
      return false;
    }
    const cv::Rect frame_rect(0, 0, depth.cols, depth.rows);
    const cv::Rect roi = bbox & frame_rect;
    if (roi.width <= 0 || roi.height <= 0) {
      return false;
    }
    const double roi_ratio = 0.5;
    const int roi_w = std::max(1, static_cast<int>(roi.width * roi_ratio));
    const int roi_h = std::max(1, static_cast<int>(roi.height * roi_ratio));
    const int center_u = roi.x + roi.width / 2;
    const int center_v = roi.y + roi.height / 2;
    cv::Rect sample_roi(center_u - roi_w / 2, center_v - roi_h / 2, roi_w,
                        roi_h);
    sample_roi &= frame_rect;
    if (sample_roi.width <= 0 || sample_roi.height <= 0) {
      return false;
    }
    const cv::Mat depth_roi = depth(sample_roi);
    const bool is_u16 = encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
                        encoding == sensor_msgs::image_encodings::MONO16 ||
                        depth.type() == CV_16UC1;
    const bool is_f32 = encoding == sensor_msgs::image_encodings::TYPE_32FC1 ||
                        depth.type() == CV_32FC1;
    if (!is_u16 && !is_f32) {
      return false;
    }
    const double roll_rad = roll * M_PI / 180.0;
    const double pitch_rad = pitch * M_PI / 180.0;
    const double install_z_m = install_z / 1000.0;
    Eigen::Matrix3d R = Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY())
                            .toRotationMatrix() *
                        Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX())
                            .toRotationMatrix();
    // 离地高度过滤：过滤地面和异常高点
    const double min_body_height_m = 0.15;
    const double max_body_height_m = 2.20;
    struct DepthPixel {
      double depth_m;
      int u;
      int v;
      double ground_x_m;
      double ground_y_m;
      double ground_z_m;
    };
    std::vector<DepthPixel> valid_pixels;
    valid_pixels.reserve(sample_roi.area());
    for (int y = 0; y < depth_roi.rows; ++y) {
      for (int x = 0; x < depth_roi.cols; ++x) {
        double d = 0.0;
        if (is_u16) {
          const uint16_t raw = depth_roi.at<uint16_t>(y, x);
          if (raw == 0) {
            continue;
          }
          d = static_cast<double>(raw) * depth_scale;
        } else {
          const float raw = depth_roi.at<float>(y, x);
          if (!std::isfinite(raw)) {
            continue;
          }
          d = static_cast<double>(raw);
        }
        const double u = sample_roi.x + x;
        const double v = sample_roi.y + y;
        // 相机坐标系：X前，Y左，Z上
        const double Xc = d;
        const double Yc = -(u - cx) * d / fx;
        const double Zc = -(v - cy) * d / fy;
        Eigen::Vector3d p_cam(Xc, Yc, Zc);
        Eigen::Vector3d p_ground =
            R * p_cam + Eigen::Vector3d(0.0, 0.0, install_z_m);
        const double Xg = p_ground.x();
        const double Yg = p_ground.y();
        const double Zg = p_ground.z();
        // 这里真正过滤地面：地面点 Zg 接近 0，不参与人体深度统计
        if (Zg < min_body_height_m || Zg > max_body_height_m ||
            d < min_depth_m || d > max_depth_m) {
          continue;
        }
        valid_pixels.push_back(DepthPixel{d, static_cast<int>(u),
                                          static_cast<int>(v), Xg, Yg, Zg});
      }
    }
    if (static_cast<int>(valid_pixels.size()) < min_valid_pixels) {
      return false;
    }
    std::vector<double> depths;
    depths.reserve(valid_pixels.size());
    for (const auto &pixel : valid_pixels) {
      depths.push_back(pixel.depth_m);
    }
    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2,
                     depths.end());
    const double median_depth_m = depths[depths.size() / 2];
    const double inlier_band_m = 0.25;
    double sum_u = 0.0;
    double sum_v = 0.0;
    double sum_ground_x = 0.0;
    double sum_ground_y = 0.0;
    double sum_ground_z = 0.0;
    int inlier_count = 0;
    for (const auto &pixel : valid_pixels) {
      if (std::abs(pixel.depth_m - median_depth_m) <= inlier_band_m) {
        sum_u += pixel.u;
        sum_v += pixel.v;
        sum_ground_x += pixel.ground_x_m;
        sum_ground_y += pixel.ground_y_m;
        sum_ground_z += pixel.ground_z_m;
        ++inlier_count;
      }
    }
    sample.depth_m = median_depth_m;
    sample.valid_count = static_cast<int>(valid_pixels.size());

    if (inlier_count < min_valid_pixels) {
      return false;
    }

    sample.u = sum_u / static_cast<double>(inlier_count);
    sample.v = sum_v / static_cast<double>(inlier_count);
    sample.ground_x_m = sum_ground_x / static_cast<double>(inlier_count);
    sample.ground_y_m = sum_ground_y / static_cast<double>(inlier_count);
    sample.ground_z_m = sum_ground_z / static_cast<double>(inlier_count);

    return true;
  }

private:
  void publish_visible_target(const std_msgs::msg::Header &header,
                              const STrack &target_track, const cv::Mat &depth,
                              const std::string &depth_encoding) {
    easygo_follow_msgs::msg::TrackedTarget msg;
    msg.header = header;
    msg.track_id = std::to_string(target_track.track_id);
    msg.visible = true;
    msg.confidence = target_track.score;

    msg.x_m = 0.0;
    msg.y_m = 0.0;
    msg.yaw_rad = 0.0;
    msg.vx_mps = 0.0;
    msg.vy_mps = 0.0;
    msg.wz_radps = 0.0;

    const cv::Rect bbox = track_to_rect(target_track);
    DepthSample sample;
    if (sample_depth_m(depth, depth_encoding, bbox, sample)) {
      // 将相机地面坐标系转为车体坐标系
      Eigen::Affine2d camera_ground_to_base_tf;
      camera_ground_to_base_tf =
          Eigen::Translation2d(install_x / 1000.0, install_y / 1000.0) *
          Eigen::Rotation2Dd(yaw * M_PI / 180.0);
      Eigen::Vector2d base_xy =
          camera_ground_to_base_tf *
          Eigen::Vector2d(sample.ground_x_m, sample.ground_y_m);

      double Xb = base_xy.x();
      double Yb = base_xy.y();

      msg.x_m = Xb;
      msg.y_m = Yb;
      msg.yaw_rad = std::atan2(Yb, Xb);
    }
    LOG(INFO) << "lgc_publish_to_motion "
          << "track_id:" << msg.track_id
          << ", visible:" << msg.visible
          << ", confidence:" << msg.confidence
          << ", x_m:" << msg.x_m
          << ", y_m:" << msg.y_m
          << ", yaw_rad:" << msg.yaw_rad
          << ", vx_mps:" << msg.vx_mps
          << ", vy_mps:" << msg.vy_mps
          << ", wz_radps:" << msg.wz_radps;
    pub_to_motion->publish(msg);
  }

  void publish_invisible_target(const std_msgs::msg::Header &header) {
    easygo_follow_msgs::msg::TrackedTarget msg;
    msg.header = header;
    msg.track_id = "";
    msg.visible = false;
    msg.confidence = 0.0f;
    msg.x_m = 0.0;
    msg.y_m = 0.0;
    msg.yaw_rad = 0.0;
    msg.vx_mps = 0.0;
    msg.vy_mps = 0.0;
    msg.wz_radps = 0.0;

    LOG(INFO) << "lgc_publish_invisible_target msg: "
          << "track_id:" << msg.track_id
          << ", visible:" << msg.visible
          << ", confidence:" << msg.confidence
          << ", x_m:" << msg.x_m
          << ", y_m:" << msg.y_m
          << ", yaw_rad:" << msg.yaw_rad
          << ", vx_mps:" << msg.vx_mps
          << ", vy_mps:" << msg.vy_mps
          << ", wz_radps:" << msg.wz_radps;
    pub_to_motion->publish(msg);
  }

  std::string track_state_to_string(int state) const {
    switch (state) {
    case TrackState::New:
      return "New";
    case TrackState::Tracked:
      return "Tracked";
    case TrackState::Lost:
      return "Lost";
    case TrackState::Removed:
      return "Removed";
    default:
      return "Unknown";
    }
  }

  void publish_tracking_debug_image(
      const std_msgs::msg::Header &header,
      const cv::Mat &src_image,
      const std::vector<STrack> &tracks,
      const std::string &stage,
      int target_track_id
  ) {
    if (!pub_tracking_image || src_image.empty()) {
      return;
    }
    cv::Mat draw_image = src_image.clone();
    const cv::Rect image_rect(0, 0, draw_image.cols, draw_image.rows);
    for (const auto &track : tracks) {
      cv::Rect box = track_to_rect(track) & image_rect;
      if (box.width <= 0 || box.height <= 0) {
        continue;
      }
      const cv::Scalar color(0,165,255);
      cv::rectangle(draw_image, box, color, 2);
      const std::string label =
          "ID:" + std::to_string(track.track_id) +
          " score:" + cv::format("%.2f", track.score) +
          " state:" + track_state_to_string(track.state) +
          " stage:" + stage;
      int baseline = 0;
      const cv::Size text_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      const int label_x = std::max(0, box.x);
      const int label_y = std::max(text_size.height + 4, box.y - 4);
      cv::rectangle(
          draw_image,
          cv::Rect(label_x, label_y - text_size.height - 4,
                   std::min(text_size.width + 4, draw_image.cols - label_x),
                   text_size.height + baseline + 4),
          color,
          cv::FILLED);
      cv::putText(draw_image, label, cv::Point(label_x + 2, label_y - 2),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    cv::putText(draw_image, "stage:" + stage + " tracks_id:" + std::to_string(target_track_id),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 255), 2);
    auto image_msg = cv_bridge::CvImage(
        header,
        sensor_msgs::image_encodings::BGR8,
        draw_image).toImageMsg();
    pub_tracking_image->publish(*image_msg);
  }

  void publish_tracking_debug_image_if_needed(
    const std_msgs::msg::Header &header,
    const cv::Mat &image,
    const std::vector<STrack> &tracks,
    bool binding_ok_this_frame) {
  std::string debug_stage;
  int target_track_id = -1;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    target_track_id = track_id;
    if (binding_ok_this_frame && track_id >= 0) {
      debug_stage = "bound";
    }else if (binding_active){
      debug_stage = "binding";
    } 
    else if (!binding_active && tracking_enabled && track_id >= 0) {
      debug_stage = "tracking";
    } else {
      return;
    }
  }
  publish_tracking_debug_image(header, image, tracks, debug_stage,target_track_id);  // 当前帧分配的ID
}

  void publish_detection_only_target(const std_msgs::msg::Header &header,
                                     const object_detect_result_list &objects,
                                     const cv::Mat &depth,
                                     const std::string &depth_encoding) {
    const int PERSON_CLASS_ID = 0;
    const object_detect_result *best_person = nullptr;
    for (int i = 0; i < objects.count; ++i) {
      const auto &obj = objects.results[i];
      if (obj.cls_id != PERSON_CLASS_ID) {
        continue;
      }
      if (!best_person || obj.prop > best_person->prop) {
        best_person = &obj;
      }
    }
    if (!best_person) {
      publish_invisible_target(header);
      return;
    }
    easygo_follow_msgs::msg::TrackedTarget msg;
    msg.header = header;
    msg.track_id = ""; // 检测模式没有稳定 track_id
    msg.visible = true;
    msg.confidence = best_person->prop;
    msg.x_m = 0.0;
    msg.y_m = 0.0;
    msg.yaw_rad = 0.0;
    msg.vx_mps = 0.0;
    msg.vy_mps = 0.0;
    msg.wz_radps = 0.0;
    const cv::Rect bbox(best_person->box.left, best_person->box.top,
                        best_person->box.right - best_person->box.left,
                        best_person->box.bottom - best_person->box.top);

    DepthSample sample;
    if (sample_depth_m(depth, depth_encoding, bbox, sample)) {
      msg.x_m = sample.ground_x_m;
    }
    pub_to_motion->publish(msg);
  }

private:
  void onReceiveRGB(const sensor_msgs::msg::Image::SharedPtr rgb_msg) {
    try {
      color_image =
          cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8)
              ->image;
      RCLCPP_INFO(this->get_logger(),"lgc_接收到彩色图像");
    } catch (const std::exception &exception) {
      RCLCPP_WARN(this->get_logger(), "Skipped rgb image message: %s",
                  exception.what());
      return;
    }

    LOG_EVERY_N(INFO,log_interval) << "lgc_进入彩色图处理流程," << "高:"<<color_image.rows << ",宽:" << color_image.cols;

    cv::Mat current_depth_image;
    std::string current_depth_encoding;
    {
      std::lock_guard<std::mutex> lock(image_mutex);
      if (!depth_image.empty()) {
        current_depth_image = depth_image;
        current_depth_encoding = depth_encoding;
      }

      if (current_depth_image.rows != color_image.rows)
      {
        LOG(INFO) << "当前深度图大小和彩色图大小不一样";
      }
    }

    LOG_EVERY_N(INFO,log_interval) << "lgc_获取到深度图," << "高:"<<current_depth_image.rows << ",宽:" << current_depth_image.cols <<",编码格式:" <<current_depth_encoding;

    object_detect_result_list objects{};
    net->detect(color_image, objects, det_conf_threshold, det_nms_threshold);

    LOG(INFO) << "lgc_objects count:" << objects.count;
    for (int i = 0; i < objects.count; ++i) {
      const auto &obj = objects.results[i];
      LOG(INFO) << "lgc_object[" << i << "] "
                << "cls_id:" << obj.cls_id
                << ", score:" << obj.prop
                << ", box:[left:" << obj.box.left
                << ", top:" << obj.box.top
                << ", right:" << obj.box.right
                << ", bottom:" << obj.box.bottom
                << "]";
    }

    // 发布检测结果
    cv::Mat draw_image = net->draw_objects(color_image, objects);
    auto image_msg = cv_bridge::CvImage(
        rgb_msg->header,
        sensor_msgs::image_encodings::BGR8,
        draw_image).toImageMsg();
    pub_detection_image->publish(*image_msg);

    bool binding_active_snapshot = false;
    bool tracking_enabled_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      binding_active_snapshot = binding_active;
      tracking_enabled_snapshot = tracking_enabled;
    }

    // 如果是检测模式
    if (!binding_active_snapshot && !tracking_enabled_snapshot) {
      LOG(INFO) << "lgc_进入检测模式";
      // publish_detection_only_target(rgb_msg->header, objects,
      //                               current_depth_image,
      //                               current_depth_encoding);
      return;
    }

    std::shared_ptr<GoalHandle> goal_for_feedback;
    std::shared_ptr<StartBinding::Feedback> feedback_msg;
    std::vector<Object> trackobj;
    std::vector<STrack> output_stracks;
    decobj_to_trackobj(objects, trackobj);
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      output_stracks = tracker->update(trackobj);
    }

    bool binding_ok_this_frame = false;

// 如果是绑定过程中
#pragma region
{
  std::lock_guard<std::mutex> lock(state_mutex);
  if (binding_active && active_goal_handle) {
    LOG(INFO) << "进入绑定状态";
    goal_for_feedback = active_goal_handle;
    feedback_msg = std::make_shared<StartBinding::Feedback>();
    if (current_depth_image.empty()) {
      binding_candidate_track_id = -1;
      binding_stable_frames = 0;
      feedback_msg->phase = StartBinding::Feedback::NO_TARGET;
      feedback_msg->code = 0;
      feedback_msg->message = "深度图为空，无法绑定";
    } else {
      std::vector<ValidCandidate> valid_candidates;
      const int raw_person_count = static_cast<int>(output_stracks.size());
      for (const auto &track : output_stracks) {
        const cv::Rect box = track_to_rect(track);
        DepthSample sample;
        if (!sample_depth_m(current_depth_image, current_depth_encoding, box,
                            sample)) {
          LOG(INFO) << "binding reject track_id=" << track.track_id
                    << ", reason=invalid_depth";
          continue;
        }
        const bool distance_ok =
            sample.ground_x_m >= min_depth_m &&
            sample.ground_x_m <= max_depth_m;
        const double real_height_m =
            static_cast<double>(box.height) * sample.depth_m / fy;
        const bool height_ok =
            real_height_m >= min_person_height_m &&
            real_height_m <= max_person_height_m;
        LOG(INFO) << "binding check track_id=" << track.track_id
                  << ", raw_person_count=" << raw_person_count
                  << ", ground_x_m=" << sample.ground_x_m
                  << ", depth_m=" << sample.depth_m
                  << ", box_height=" << box.height
                  << ", real_height_m=" << real_height_m
                  << ", distance_ok=" << distance_ok
                  << ", height_ok=" << height_ok;
        if (!distance_ok || !height_ok) {
          continue;
        }
        valid_candidates.push_back(
            ValidCandidate{&track, box, sample, real_height_m});
      }
      const int valid_person_count =
          static_cast<int>(valid_candidates.size());
      if (valid_person_count == 1) {
        const STrack &candidate = *valid_candidates.front().track;
        LOG(INFO) << "binding valid_person_count=" << valid_person_count
                  << ", candidate_track_id=" << candidate.track_id
                  << ", stable_frames=" << binding_stable_frames;
        if (binding_candidate_track_id == candidate.track_id) {
          ++binding_stable_frames;
        } else {
          binding_candidate_track_id = candidate.track_id;
          binding_stable_frames = 1;
        }
        feedback_msg->phase = StartBinding::Feedback::WAITING;
        feedback_msg->code = static_cast<uint16_t>(binding_stable_frames);
        feedback_msg->message =
            "单目标稳定中:" + std::to_string(binding_stable_frames) +
            "/" + std::to_string(kRequiredStableFrames);
        if (binding_stable_frames >= kRequiredStableFrames) {
          track_id = candidate.track_id;
          binding_active = false;
          binding_completed = true;
          tracking_enabled = true;
          binding_ok_this_frame = true;
          feedback_msg->phase = StartBinding::Feedback::BOUND;
          feedback_msg->code = StartBinding::Feedback::BOUND;
          feedback_msg->message =
              "绑定成功,track_id=" + std::to_string(track_id);
        }
      } else if (valid_person_count == 0) {
        binding_candidate_track_id = -1;
        binding_stable_frames = 0;
        feedback_msg->phase = StartBinding::Feedback::NO_TARGET;
        feedback_msg->code = 0;
        if (raw_person_count == 0) {
          feedback_msg->message = "未检测到目标";
        } else {
          feedback_msg->message = "未检测到符合距离和身高条件的目标";
        }
      } else {
        binding_candidate_track_id = -1;
        binding_stable_frames = 0;
        feedback_msg->phase = StartBinding::Feedback::MULTI_TARGET;
        feedback_msg->code = static_cast<uint16_t>(valid_person_count);
        feedback_msg->message = "检测到多个有效目标，无法绑定";
      }
    }
  }
}
if (goal_for_feedback && feedback_msg) {
  goal_for_feedback->publish_feedback(feedback_msg);
}
#pragma endregion

publish_tracking_debug_image_if_needed(rgb_msg->header, color_image, output_stracks, binding_ok_this_frame);

// 如果是跟踪状态
#pragma region
    int track_id_snapshot = -1;
    bool should_track = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      should_track = !binding_active && tracking_enabled && track_id >= 0;
      track_id_snapshot = track_id;
    } // 锁释放

    if (should_track) {
      LOG(INFO) << "进入跟踪状态";
      bool found = false;
      for (const auto &track : output_stracks) {
        if (track.track_id == track_id_snapshot) {
          publish_visible_target(rgb_msg->header, track, current_depth_image,
                                 current_depth_encoding);
          publish_perception_event(
              rgb_msg->header,
              easygo_follow_msgs::msg::PerceptionEvent::TARGET_ACQUIRED, 0,
              "target acquired");
          found = true;
          break;
        }
      }
      if (!found) {
        publish_invisible_target(rgb_msg->header);
        publish_perception_event(
            rgb_msg->header,
            easygo_follow_msgs::msg::PerceptionEvent::TARGET_LOST, 1,
            "target lost");
      }
    }

#pragma endregion
  }

  void onReceiveDepth(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
    try {
      cv::Mat converted =
          cv_bridge::toCvCopy(depth_msg, depth_msg->encoding)->image;
      std::lock_guard<std::mutex> lock(image_mutex);
      depth_image = converted;
      depth_encoding = depth_msg->encoding;
      RCLCPP_INFO(this->get_logger(),"接收到深度图片");
      LOG(INFO) << "lgc" << "接收到深度图片";
    } catch (const std::exception &exception) {
      RCLCPP_WARN(this->get_logger(), "Skipped depth image message: %s",
                  exception.what());
    }
  }

  // 停止跟踪
  void stop_tracking(
      const std::shared_ptr<easygo_follow_msgs::srv::StopTracking::Request>
          request,
      std::shared_ptr<easygo_follow_msgs::srv::StopTracking::Response>
          response) {
    (void)request;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      // 在跟踪的时候停止
      if (!binding_active && tracking_enabled) {
        LOG(INFO) << "在跟踪的时候停止";
        track_id = -1;
        tracking_enabled = false;
        binding_active = false;
        binding_completed = false;
        binding_stable_frames = 0;
        binding_candidate_track_id = -1;
      }

      // 在绑定的时候停止
      if (binding_active && !tracking_enabled) {
        LOG(INFO) << "在绑定的时候停止";
        track_id = -1;
        tracking_enabled = false;
        binding_active = false;
        binding_completed = false;
        binding_stable_frames = 0;
        binding_candidate_track_id = -1;
        stop_requested = true;
      }

      // 在检测的时候停止
      if (!binding_active && !tracking_enabled) {
        LOG(INFO) << "在检测的时候停止";
        track_id = -1;
        binding_active = false;
        binding_completed = false;
        binding_stable_frames = 0;
        binding_candidate_track_id = -1;
      }
    }
    reset_track();
    response->result.accepted = true;
    response->result.code = 0;
    response->result.message = "停止跟踪成功，已切换为仅检测输出";
  }

  // 开始绑定消息
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const easygo_follow_msgs::action::StartBinding::Goal>
          goal_request) {
    (void)uuid;
    (void)goal_request;

    RCLCPP_INFO(this->get_logger(), "Server: Received goal request");

    // 原子地检查并占用执行权
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (binding_active || tracking_enabled) {
        RCLCPP_WARN(this->get_logger(),
                    "Server: Another goal is running, reject new goal");
        return rclcpp_action::GoalResponse::REJECT;
      }
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  // "绑定取消响应"
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandle> goal_handle_canceled) {
    (void)goal_handle_canceled;
    RCLCPP_INFO(this->get_logger(),
                "Server: Received request to cancel action");

    return rclcpp_action::CancelResponse::ACCEPT;
  }

  // 开始绑定
  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle_accepted) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      active_goal_handle = goal_handle_accepted;
      tracking_enabled = false;
      binding_candidate_track_id = -1;
      binding_stable_frames = 0;
      stop_requested = false;
      binding_active = true;
      binding_completed = false;
      binding_start_time = this->now();
      track_id = -1;
    }
    auto feedback = std::make_shared<StartBinding::Feedback>();
    feedback->phase = StartBinding::Feedback::WAITING;
    feedback->code = 0;
    feedback->message = "等待目标进入视野";
    goal_handle_accepted->publish_feedback(feedback);

    // 在线程中执行动作过程
    using namespace std::placeholders;
    std::thread{std::bind(&PersonTrackNode::execute, this, _1),
                goal_handle_accepted}
        .detach();
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle) {
    auto result =
        std::make_shared<easygo_follow_msgs::action::StartBinding::Result>();
    RCLCPP_INFO(this->get_logger(), "Server: Executing binding");
    rclcpp::Rate rate(10.0);

    // 开始绑定
    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        reset_all_tracking();
        result->success = false;
        result->code = 0;
        result->message = "绑定已取消";
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Server: Goal canceled");
        return;
      }

      bool completed = false;
      bool should_stop = false;
      int bound_track_id = -1;
      rclcpp::Time start_time(0, 0, RCL_ROS_TIME);
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (active_goal_handle != goal_handle) {
          return;
        }

        bound_track_id = track_id;
        start_time = binding_start_time;
        should_stop = stop_requested;
        completed = binding_completed;
      }

      // 停止跟踪
      if (should_stop) {
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          stop_requested = false;
          active_goal_handle.reset();
        }
        result->success = false;
        result->code = 0;
        result->message = "绑定被 stop_tracking 服务终止";
        goal_handle->canceled(result);
        RCLCPP_WARN(this->get_logger(),
                    "Server: Goal aborted by stop_tracking");
        return;
      }

      // 绑定完成
      if (completed && bound_track_id >= 0) {
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          binding_completed = true;
          binding_active = false;
          tracking_enabled = true;
          binding_candidate_track_id = -1;
          binding_stable_frames = 0;
          active_goal_handle.reset();
        }
        result->success = true;
        result->code = StartBinding::Feedback::BOUND;
        result->message = "绑定成功,track_id=" + std::to_string(bound_track_id);
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(),
                    "Server: Goal succeeded, bind track_id=%d", bound_track_id);
        return;
      }

      // 绑定超时
      if ((this->now() - start_time).seconds() > kBindTimeoutSec) {
        reset_all_tracking();
        result->success = false;
        result->code = 0;
        result->message = "绑定超时";
        goal_handle->abort(result);
        RCLCPP_WARN(this->get_logger(),
                    "Server: Goal aborted, binding timeout");
        return;
      }
      rate.sleep();
    }

    reset_all_tracking();
    result->success = false;
    result->code = 0;
    result->message = "ROS 已退出";
    goal_handle->abort(result);
  }

private:
  // 订阅彩色图像
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      subscription_rgb_image;
  std::string rgb_topic;
  cv::Mat color_image;

  // 订阅深度图像
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      subscription_depth_image;
  std::string depth_topic;
  cv::Mat depth_image;
  std::string depth_encoding;
  double depth_scale = 0.001;

  // 发送结果话题至运控
  std::string tracked_target_topic;
  rclcpp::Publisher<easygo_follow_msgs::msg::TrackedTarget>::SharedPtr
      pub_to_motion;

  // 发送结果给app
  std::string perception_event_topic;
  rclcpp::Publisher<easygo_follow_msgs::msg::PerceptionEvent>::SharedPtr
      pub_to_app;

  // 开始绑定
  std::string action_server_name;
  rclcpp_action::Server<easygo_follow_msgs::action::StartBinding>::SharedPtr
      binding_action;

  // 停止跟踪
  std::string stop_track_server_name;
  rclcpp::Service<easygo_follow_msgs::srv::StopTracking>::SharedPtr
      stop_track_server;

  // 检测器和跟踪器
  std::string model_path;
  std::shared_ptr<Yolov8Detector> net;
  std::shared_ptr<BYTETracker> tracker;
  double det_conf_threshold = 0.25;
  double det_nms_threshold = 0.45;
  double min_depth_m = 0.5;
  double max_depth_m = 5;
  double min_person_height_m = 0.8;
  double max_person_height_m = 2.2;
  int min_valid_pixels = 10;
  int track_id = -1;

  static constexpr int kRequiredStableFrames = 20; // 连续20帧即绑定成功
  static constexpr double kBindTimeoutSec = 60;    // 超时即绑定失败
  std::mutex
      state_mutex; // 状态锁，并发入口包括:onReceiveRGB()/onReceiveDepth()/handle_goal()
                   // / handle_cancel() / handle_accepted()/execute()
  std::mutex tracker_mutex; // 跟踪锁:保证update和reset的顺序
  std::mutex image_mutex; // 防止depth_image的同时读(onReceiveRGB)写(onReceiveDepth)

  rclcpp::Time binding_start_time =
      rclcpp::Time(0, 0, RCL_ROS_TIME); // 开始绑定时间

  int binding_stable_frames = 0;       // 绑定保持帧数
  int binding_candidate_track_id = -1; // 当前的绑定ID
  bool binding_active = false;         // 是否在绑定的过程中
  bool binding_completed = false;      // 是否在绑定已完成
  bool stop_requested = false; // 停止跟踪/绑定请求，由 execute() 统一结束 action
  bool tracking_enabled = false; // 跟踪模式

  std::shared_ptr<GoalHandle> active_goal_handle; // 发送反馈消息的句柄

  // 相机内参
  double fx = 481.0250244140625;
  double fy = 481.12005615234375;
  double cx = 323.1153259277344;
  double cy = 238.83233642578125;

  double install_x = 360;
  double install_y = 0;
  double install_z = 124;
  double roll = 0;
  double pitch = -18;
  double yaw = 0;

  // 调试代码
  std::string detect_result_image_topic;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_detection_image;
  int log_interval = 100;
  std::string track_result_image_topic;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_tracking_image;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PersonTrackNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

// 检测模式: !binding_active && !tracking_enabled
// 绑定模式: binding_active
// 跟踪模式: tracking_enabled && track_id >= 0