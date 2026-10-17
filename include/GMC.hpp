#pragma once
#include <opencv2/core.hpp>

// Port of ultralytics/trackers/utils/gmc.py, "sparseOptFlow" method only
// (botsort.yaml's default gmc_method). Estimates the frame-to-frame camera
// motion as a 2x3 affine warp, used to compensate track positions before
// association so a panning/moving camera doesn't look like object motion.
class GMC {
public:
    explicit GMC(int downscale = 2);

    // raw_frame: current BGR frame. Returns a 2x3 CV_64F affine warp
    // (identity on the first call, or whenever too few points match).
    cv::Mat apply(const cv::Mat& rawFrame);

private:
    int downscale_;
    cv::Mat prevFrameGray_;
    std::vector<cv::Point2f> prevKeyPoints_;
    bool initialized_ = false;
};
