#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <opencv2/core.hpp>
#include "STrack.hpp"
#include "KalmanFilterXYWH.hpp"
#include "GMC.hpp"

// One raw detection for a single frame, as produced by Detector.
struct Detection {
    cv::Rect2f tlwh; // (x, y, w, h)
    float score;
    int cls;
};

// Defaults below mirror ultralytics' default botsort.yaml exactly, since that
// is the tracker config Vehicle-and-Pedestrian-Traffic-Analysis-System.py runs with
// (model.track(..., tracker="botsort.yaml")). Notably with_reid is False by
// default there too — ReID plumbing is fully implemented here (per request)
// but stays inactive unless you supply an encoder via setReidEncoder().
struct BotSortParams {
    double trackHighThresh = 0.5;
    double trackLowThresh = 0.1;
    double newTrackThresh = 0.6;
    int trackBuffer = 30;
    double matchThresh = 0.8;
    bool fuseScore = true;
    double proximityThresh = 0.5;
    double appearanceThresh = 0.25;
    bool withReid = false;
    double frameRate = 30.0;
};

// Port of ultralytics/trackers/bot_sort.py (BOTSORT) + byte_tracker.py
// (BYTETracker.update cascade), the tracker actually driving
// model.track(..., tracker="botsort.yaml") in the Python script.
class BotSort {
public:
    explicit BotSort(const BotSortParams& params);

    // Optional ReID hook: given the frame and a detection box (tlwh), return
    // an embedding row-vector (CV_32F/CV_64F, any length, need not be
    // pre-normalized). Only ever called when params.withReid is true.
    void setReidEncoder(std::function<cv::Mat(const cv::Mat&, const cv::Rect2f&)> encoder) {
        reidEncoder_ = std::move(encoder);
    }

    // Advances the tracker by one frame. `frame` is used for GMC (and ReID
    // crops, if enabled). Returns the currently active, activated tracks.
    std::vector<std::shared_ptr<STrack>> update(const std::vector<Detection>& detections,
                                                 const cv::Mat& frame);

private:
    std::vector<std::shared_ptr<STrack>> makeTracks(const std::vector<Detection>& dets,
                                                      const cv::Mat& frame) const;
    std::vector<std::vector<double>> getDists(const std::vector<STrack*>& tracks,
                                               const std::vector<STrack*>& dets) const;

    BotSortParams params_;
    KalmanFilterXYWH kalmanFilter_;
    GMC gmc_;
    int frameId_ = 0;
    int maxTimeLost_;
    std::function<cv::Mat(const cv::Mat&, const cv::Rect2f&)> reidEncoder_;

    std::vector<std::shared_ptr<STrack>> trackedStracks_;
    std::vector<std::shared_ptr<STrack>> lostStracks_;
    std::vector<std::shared_ptr<STrack>> removedStracks_;
};
