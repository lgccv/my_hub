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
	BYTETracker(float track_thresh=0.25, float high_thresh=0.25, float match_thresh=0.8, int max_time_lost=30);
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
};
