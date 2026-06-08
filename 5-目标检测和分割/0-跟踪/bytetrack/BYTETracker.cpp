#include "BYTETracker.h"
#include <fstream>
#include "glog/logging.h"


BYTETracker::BYTETracker(float track_thresh, float high_thresh, float match_thresh, int max_time_lost)
{
	this->track_thresh = track_thresh;
	this->high_thresh = high_thresh;
	this->match_thresh = match_thresh;

	this->frame_id = 0;
	this->max_time_lost = max_time_lost;

	LOG(INFO) << "Init ByteTrack!";

}

BYTETracker::~BYTETracker()
{
}

void BYTETracker::reset(bool reset_id)
{
    this->frame_id = 0;
    this->tracked_stracks.clear();
    this->lost_stracks.clear();
    this->removed_stracks.clear();
    // 重新初始化 KalmanFilter，避免残留状态
    this->kalman_filter = byte_kalman::KalmanFilter();
    if (reset_id)
    {
        STrack::reset_id_count();
    }
    LOG(INFO) << "ByteTracker reset, reset_id=" << reset_id;
}

std::string BYTETracker::get_track_status(int track_id) const
{
    for (const auto& track : this->tracked_stracks)
    {
        if (track.track_id == track_id)
        {
            return "tracked";
        }
    }
    for (const auto& track : this->lost_stracks)
    {
        if (track.track_id == track_id)
        {
            return "lost";
        }
    }
    for (const auto& track : this->removed_stracks)
    {
        if (track.track_id == track_id)
        {
            return "removed";
        }
    }
    return "not_found";
}

 std::vector<STrack> BYTETracker::update(const  std::vector<Object>& objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	 std::vector<STrack> activated_stracks;
	 std::vector<STrack> refind_stracks;
	 std::vector<STrack> removed_stracks;
	 std::vector<STrack> lost_stracks;
	 std::vector<STrack> detections;
	 std::vector<STrack> detections_low;

	 std::vector<STrack> detections_cp;
	 std::vector<STrack> tracked_stracks_swap;
	 std::vector<STrack> resa, resb;
	 std::vector<STrack> output_stracks;

	 std::vector<STrack*> unconfirmed;
	 std::vector<STrack*> tracked_stracks;
	 std::vector<STrack*> strack_pool;
	 std::vector<STrack*> r_tracked_stracks;

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			std::vector<float> tlbr_;
			tlbr_.resize(4);
            tlbr_[0] = objects[i].box.x;
            tlbr_[1] = objects[i].box.y;
            tlbr_[2] = objects[i].box.x + objects[i].box.width;
            tlbr_[3] = objects[i].box.y + objects[i].box.height;

			float score = objects[i].score;
			int classId = objects[i].classId;

			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score,classId);
			if (score >= track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
			
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU //////////////////
	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
	STrack::multi_predict(strack_pool, this->kalman_filter);

    // 传入 BYTETrack 后，先把高分检测框按 IoU > 0.15 合并。
    // 后面 dists 使用的是合并后的 detections。
    // detections = merge_overlapping_detections(detections, 0.15F);

    MergedDetectionsResult merge_result = merge_overlapping_detections(detections, 0.15F);
    detections = merge_result.detections;
    std::vector<bool> detection_is_merged = merge_result.is_merged;


	std::vector< std::vector<float> > dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);
    print_dists_matrix(dists, strack_pool, detections, "first association dists");

	// 定制逻辑
    //apply_fusion_box_protection(dists,strack_pool,detections);
    apply_fusion_box_protection(dists,strack_pool,detections,detection_is_merged);
	// 定制逻辑结束

    print_dists_matrix(dists, strack_pool, detections, "first association dist2");
	std::vector< std::vector<int> > matches;
	std::vector<int> u_track, u_detection;
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());
	
	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	 std::vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	//std::cout << activated_stracks.size() << std::endl;

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}
	
	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());
	
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}
	return output_stracks;
}


void BYTETracker::print_dists_matrix(
    const std::vector<std::vector<float>> &dists,
    const std::vector<STrack*> &strack_pool,
    const std::vector<STrack> &detections,
    const std::string &name) const
{
    std::cout << "===== " << name << " =====" << std::endl;

    std::cout << "rows = " << dists.size()
              << ", cols = "
              << (dists.empty() ? 0 : dists[0].size())
              << std::endl;

    if (dists.empty() || detections.empty())
    {
        std::cout << "(empty)" << std::endl;
        std::cout << "====================" << std::endl;
        return;
    }

    // 第一行：检测框分数
    std::cout << std::setw(10) << "track_id";
    for (int j = 0; j < static_cast<int>(detections.size()); ++j)
    {
        std::cout << std::setw(10)
                  << std::fixed
                  << std::setprecision(3)
                  << detections[j].score;
    }

    std::cout << std::endl;

    // 后续每一行：第一列是轨迹 ID，后面是 dists 值
    for (int i = 0; i < static_cast<int>(dists.size()); ++i)
    {
        int track_id = -1;

        if (i < static_cast<int>(strack_pool.size()) && strack_pool[i])
        {
            track_id = strack_pool[i]->track_id;
        }

        std::cout << std::setw(10) << track_id;

        for (int j = 0; j < static_cast<int>(dists[i].size()); ++j)
        {
            std::cout << std::setw(10)
                      << std::fixed
                      << std::setprecision(4)
                      << dists[i][j];
        }

        std::cout << std::endl;
    }

    std::cout << "====================" << std::endl;
}

float BYTETracker::calc_iou_tlwh(
    const std::vector<float> &a_tlwh,
    const std::vector<float> &b_tlwh) const
{
    if (a_tlwh.size() < 4 || b_tlwh.size() < 4)
    {
        return 0.0F;
    }

    float ax1 = a_tlwh[0];
    float ay1 = a_tlwh[1];
    float ax2 = a_tlwh[0] + a_tlwh[2];
    float ay2 = a_tlwh[1] + a_tlwh[3];

    float bx1 = b_tlwh[0];
    float by1 = b_tlwh[1];
    float bx2 = b_tlwh[0] + b_tlwh[2];
    float by2 = b_tlwh[1] + b_tlwh[3];

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0F, inter_x2 - inter_x1);
    float inter_h = std::max(0.0F, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = std::max(0.0F, a_tlwh[2]) * std::max(0.0F, a_tlwh[3]);
    float area_b = std::max(0.0F, b_tlwh[2]) * std::max(0.0F, b_tlwh[3]);

    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0F)
    {
        return 0.0F;
    }

    return inter_area / union_area;
}

int BYTETracker::find_protected_row(
    const std::vector<STrack*> &strack_pool) const
{
    if (protected_track_id <= 0)
    {
        return -1;
    }

    for (int i = 0; i < static_cast<int>(strack_pool.size()); ++i)
    {
        if (!strack_pool[i])
        {
            continue;
        }

        if (strack_pool[i]->track_id == protected_track_id)
        {
            return i;
        }
    }

    return -1;
}

std::vector<std::vector<int>> BYTETracker::build_detection_iou_groups(
    const std::vector<STrack> &detections,
    float iou_thresh) const
{
    std::vector<std::vector<int>> groups;
    std::vector<bool> visited(detections.size(), false);

    for (int i = 0; i < static_cast<int>(detections.size()); ++i)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<int> group;
        std::vector<int> stack;

        visited[i] = true;
        stack.push_back(i);

        while (!stack.empty())
        {
            int cur = stack.back();
            stack.pop_back();
            group.push_back(cur);

            for (int j = 0; j < static_cast<int>(detections.size()); ++j)
            {
                if (visited[j])
                {
                    continue;
                }

                float det_iou =
                    calc_iou_tlwh(detections[cur].tlwh, detections[j].tlwh);

                if (det_iou > iou_thresh)
                {
                    visited[j] = true;
                    stack.push_back(j);
                }
            }
        }

        groups.push_back(group);
    }

    return groups;
}

STrack BYTETracker::merge_detection_group(
    const std::vector<STrack> &detections,
    const std::vector<int> &group) const
{
    int first_idx = group[0];

    float x1 = detections[first_idx].tlwh[0];
    float y1 = detections[first_idx].tlwh[1];
    float x2 = detections[first_idx].tlwh[0] + detections[first_idx].tlwh[2];
    float y2 = detections[first_idx].tlwh[1] + detections[first_idx].tlwh[3];

    float best_score = detections[first_idx].score;
    int best_class_id = detections[first_idx].classId;

    for (int det_idx : group)
    {
        const std::vector<float> &tlwh = detections[det_idx].tlwh;

        x1 = std::min(x1, tlwh[0]);
        y1 = std::min(y1, tlwh[1]);
        x2 = std::max(x2, tlwh[0] + tlwh[2]);
        y2 = std::max(y2, tlwh[1] + tlwh[3]);

        if (detections[det_idx].score > best_score)
        {
            best_score = detections[det_idx].score;
            best_class_id = detections[det_idx].classId;
        }
    }

    std::vector<float> merged_tlwh = {x1, y1, x2 - x1, y2 - y1};

    return STrack(merged_tlwh, best_score, best_class_id);
}

// std::vector<STrack> BYTETracker::merge_overlapping_detections(
//     const std::vector<STrack> &detections,
//     float iou_thresh) const
// {
//     std::vector<STrack> merged_detections;

//     if (detections.empty())
//     {
//         return merged_detections;
//     }

//     std::vector<std::vector<int>> groups =
//         build_detection_iou_groups(detections, iou_thresh);

//     for (const std::vector<int> &group : groups)
//     {
//         if (group.empty())
//         {
//             continue;
//         }

//         if (group.size() == 1)
//         {
//             merged_detections.push_back(detections[group[0]]);
//             continue;
//         }

//         merged_detections.push_back(
//             merge_detection_group(detections, group));
//     }

//     return merged_detections;
// }

bool BYTETracker::detection_intersects_other_detections(
    const std::vector<STrack> &detections,
    int det_col,
    float iou_thresh) const
{
    if (det_col < 0 || det_col >= static_cast<int>(detections.size()))
    {
        return false;
    }

    for (int other_col = 0; other_col < static_cast<int>(detections.size()); ++other_col)
    {
        if (other_col == det_col)
        {
            continue;
        }

        float overlap_iou =
            calc_iou_tlwh(detections[det_col].tlwh, detections[other_col].tlwh);

        if (overlap_iou > iou_thresh)
        {
            return true;
        }
    }

    return false;
}

bool BYTETracker::detection_intersects_multi_tracks_with_protected(
    const STrack &detection,
    const std::vector<STrack*> &strack_pool,
    int protected_row,
    float track_iou_thresh) const
{
    if (protected_row < 0)
    {
        return false;
    }

    int intersect_track_count = 0;
    bool contains_protected_track = false;

    for (int row = 0; row < static_cast<int>(strack_pool.size()); ++row)
    {
        if (!strack_pool[row])
        {
            continue;
        }

        float track_iou =
            calc_iou_tlwh(detection.tlwh, strack_pool[row]->tlwh);

        if (track_iou <= track_iou_thresh)
        {
            continue;
        }

        ++intersect_track_count;

        if (row == protected_row)
        {
            contains_protected_track = true;
        }
    }

    return intersect_track_count >= 2 && contains_protected_track;
}

// void BYTETracker::apply_fusion_box_protection(
//     std::vector<std::vector<float>> &dists,
//     const std::vector<STrack*> &strack_pool,
//     const std::vector<STrack> &detections) const
// {
//     int protected_row = find_protected_row(strack_pool);

//     if (protected_row < 0 || dists.empty() || detections.empty())
//     {
//         return;
//     }

//     for (int det_col = 0; det_col < static_cast<int>(detections.size()); ++det_col)
//     {
//         bool hit_multi_tracks =
//             detection_intersects_multi_tracks_with_protected(
//                 detections[det_col],
//                 strack_pool,
//                 protected_row,
//                 0.15F);

//         if (!hit_multi_tracks)
//         {
//             continue;
//         }

//         bool has_other_detection_overlap =
//             detection_intersects_other_detections(
//                 detections,
//                 det_col,
//                 0.0F);

//         if (has_other_detection_overlap)
//         {
//             continue;
//         }

//         if (det_col < 0 ||
//             det_col >= static_cast<int>(dists[protected_row].size()))
//         {
//             continue;
//         }

//         dists[protected_row][det_col] = protected_cost_override;
//     }
// }


void BYTETracker::apply_fusion_box_protection(std::vector<std::vector<float>> &dists,const std::vector<STrack*> &strack_pool,const std::vector<STrack> &detections,const std::vector<bool> &detection_is_merged) const
{
    int protected_row = find_protected_row(strack_pool);

    if (protected_row < 0 || dists.empty() || detections.empty())
    {
        return;
    }

    for (int det_col = 0; det_col < static_cast<int>(detections.size()); ++det_col) // 开始遍历每一个检测框
    {
        if (det_col >= static_cast<int>(detection_is_merged.size()))
        {
            continue;
        }

        if (det_col >= static_cast<int>(dists[protected_row].size()))
        {
            continue;
        }

        bool hit_multi_tracks_015 =
            detection_intersects_multi_tracks_with_protected(
                detections[det_col],
                strack_pool,
                protected_row,
                0.15F);  // 这个同时压住了多条轨迹，而且其中一条是保护轨迹

        if (!hit_multi_tracks_015)
        {
            continue;
        }

        if (detection_is_merged[det_col])   // 如果是合成框
        {
            dists[protected_row][det_col] = protected_cost_override;
            continue;
        }

        bool has_other_detection_overlap =
            detection_intersects_other_detections(
                detections,
                det_col,
                0.0F);

        if (has_other_detection_overlap)
        {
            continue;
        }

        bool hit_multi_tracks_025 =
            detection_intersects_multi_tracks_with_protected(
                detections[det_col],
                strack_pool,
                protected_row,
                0.25F);

        if (hit_multi_tracks_025)
        {
            dists[protected_row][det_col] = protected_cost_override;
        }
    }
}


BYTETracker::MergedDetectionsResult BYTETracker::merge_overlapping_detections(
    const std::vector<STrack> &detections,
    float iou_thresh) const
{
    MergedDetectionsResult result;

    if (detections.empty())
    {
        return result;
    }

    std::vector<std::vector<int>> groups =
        build_detection_iou_groups(detections, iou_thresh);

    for (const std::vector<int> &group : groups)
    {
        if (group.empty())
        {
            continue;
        }

        if (group.size() == 1)
        {
            result.detections.push_back(detections[group[0]]);
            result.is_merged.push_back(false);
            continue;
        }

        result.detections.push_back(
            merge_detection_group(detections, group));
        result.is_merged.push_back(true);
    }

    return result;
}