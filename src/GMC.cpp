#include "GMC.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>

GMC::GMC(int downscale) : downscale_(std::max(1, downscale)) {}

cv::Mat GMC::apply(const cv::Mat& rawFrame) {
    cv::Mat H = cv::Mat::eye(2, 3, CV_64F);

    int width = rawFrame.cols, height = rawFrame.rows;
    cv::Mat gray;
    cv::cvtColor(rawFrame, gray, cv::COLOR_BGR2GRAY);
    if (downscale_ > 1) {
        cv::resize(gray, gray, cv::Size(width / downscale_, height / downscale_));
    }

    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(gray, keypoints, /*maxCorners=*/1000, /*qualityLevel=*/0.01,
                             /*minDistance=*/1, cv::noArray(), /*blockSize=*/3,
                             /*useHarrisDetector=*/false, /*k=*/0.04);

    if (!initialized_ || prevKeyPoints_.empty()) {
        initialized_ = true;
        prevFrameGray_ = gray.clone();
        prevKeyPoints_ = keypoints;
        return H;
    }

    std::vector<cv::Point2f> matched;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevFrameGray_, gray, prevKeyPoints_, matched, status, err);

    std::vector<cv::Point2f> prevPts, currPts;
    prevPts.reserve(status.size());
    currPts.reserve(status.size());
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            prevPts.push_back(prevKeyPoints_[i]);
            currPts.push_back(matched[i]);
        }
    }

    if (prevPts.size() > 4) {
        cv::Mat inliers;
        cv::Mat est = cv::estimateAffinePartial2D(prevPts, currPts, inliers, cv::RANSAC);
        if (!est.empty()) {
            est.convertTo(H, CV_64F);
            if (downscale_ > 1) {
                H.at<double>(0, 2) *= downscale_;
                H.at<double>(1, 2) *= downscale_;
            }
        }
    }

    prevFrameGray_ = gray.clone();
    prevKeyPoints_ = keypoints;
    return H;
}
