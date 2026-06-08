#pragma once

#include "STrack.h"
#include <string>

struct Object
{
    int classId;
    float score;
    cv::Rect_<float> box;
};

class BYTETracker
{
public:
	//BYTETracker(int frame_rate = 30, int track_buffer = 30);
	BYTETracker(float track_thresh=0.5, float high_thresh=0.5, float match_thresh=0.95, int max_time_lost=100);
	~BYTETracker();

	std::vector<STrack> update(const  std::vector<Object>& objects);
    cv::Scalar get_color(int idx);
	void reset(bool reset_id = false);

	std::string get_track_status(int track_id) const;

private:
	 std::vector<STrack*> joint_stracks( std::vector<STrack*> &tlista,  std::vector<STrack> &tlistb);
	 std::vector<STrack> joint_stracks( std::vector<STrack> &tlista,  std::vector<STrack> &tlistb);

	 std::vector<STrack> sub_stracks( std::vector<STrack> &tlista,  std::vector<STrack> &tlistb);
	void remove_duplicate_stracks( std::vector<STrack> &resa,  std::vector<STrack> &resb,  std::vector<STrack> &stracksa,  std::vector<STrack> &stracksb);

	void linear_assignment( std::vector< std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		 std::vector< std::vector<int> > &matches,  std::vector<int> &unmatched_a,  std::vector<int> &unmatched_b);
	 std::vector< std::vector<float> > iou_distance( std::vector<STrack*> &atracks,  std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	 std::vector< std::vector<float> > iou_distance( std::vector<STrack> &atracks,  std::vector<STrack> &btracks);
	 std::vector< std::vector<float> > ious( std::vector< std::vector<float> > &atlbrs,  std::vector< std::vector<float> > &btlbrs);

	double lapjv(const  std::vector< std::vector<float> > &cost,  std::vector<int> &rowsol,  std::vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;

	void print_dists_matrix(
		const std::vector<std::vector<float>> &dists,
		const std::vector<STrack*> &strack_pool,
		const std::vector<STrack> &detections,
		const std::string &name) const;

	int protected_track_id = 1;
	float protected_cost_override = 1.0F;

	float calc_iou_tlwh(
		const std::vector<float> &a_tlwh,
		const std::vector<float> &b_tlwh) const;

	int find_protected_row(
		const std::vector<STrack*> &strack_pool) const;

	std::vector<std::vector<int>> build_detection_iou_groups(
		const std::vector<STrack> &detections,
		float iou_thresh) const;

	STrack merge_detection_group(
		const std::vector<STrack> &detections,
		const std::vector<int> &group) const;

	// std::vector<STrack> merge_overlapping_detections(
	// 	const std::vector<STrack> &detections,
	// 	float iou_thresh) const;

	struct MergedDetectionsResult
	{
    	std::vector<STrack> detections;
    	std::vector<bool> is_merged;
	};

	MergedDetectionsResult merge_overlapping_detections(const std::vector<STrack> &detections,float iou_thresh) const;

	bool detection_intersects_other_detections(
		const std::vector<STrack> &detections,
		int det_col,
		float iou_thresh) const;

	bool detection_intersects_multi_tracks_with_protected(
		const STrack &detection,
		const std::vector<STrack*> &strack_pool,
		int protected_row,
		float track_iou_thresh) const;

	// void apply_fusion_box_protection(
	// 	std::vector<std::vector<float>> &dists,
	// 	const std::vector<STrack*> &strack_pool,
	// 	const std::vector<STrack> &detections) const;
	void apply_fusion_box_protection(std::vector<std::vector<float>> &dists,const std::vector<STrack*> &strack_pool,const std::vector<STrack> &detections,const std::vector<bool> &detection_is_merged) const;


};
