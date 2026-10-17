#pragma once
#include <opencv2/core.hpp>

// 8-state constant-velocity Kalman filter over (x, y, w, h) center-form boxes,
// port of ultralytics/trackers/utils/kalman_filter.py: KalmanFilterXYAH's base
// math specialized to BoT-SORT's KalmanFilterXYWH (measures w, h directly
// instead of an aspect ratio). State = [x, y, w, h, vx, vy, vw, vh]^T.
class KalmanFilterXYWH {
public:
    KalmanFilterXYWH();

    // measurement = (x, y, w, h) center-form box. Returns (mean[8x1], covariance[8x8]).
    void initiate(const cv::Vec4f& measurement, cv::Mat& mean, cv::Mat& covariance) const;

    void predict(cv::Mat& mean, cv::Mat& covariance) const;

    void project(const cv::Mat& mean, const cv::Mat& covariance,
                 cv::Mat& projMean, cv::Mat& projCovariance) const;

    void update(cv::Mat& mean, cv::Mat& covariance, const cv::Vec4f& measurement) const;

private:
    cv::Mat motionMat_;  // 8x8
    cv::Mat updateMat_;  // 4x8
    double stdWeightPosition_;
    double stdWeightVelocity_;
};
