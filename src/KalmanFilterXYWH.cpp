#include "KalmanFilterXYWH.hpp"

KalmanFilterXYWH::KalmanFilterXYWH()
    : stdWeightPosition_(1.0 / 20), stdWeightVelocity_(1.0 / 160) {
    const int ndim = 4;
    const double dt = 1.0;
    motionMat_ = cv::Mat::eye(2 * ndim, 2 * ndim, CV_64F);
    for (int i = 0; i < ndim; i++)
        motionMat_.at<double>(i, ndim + i) = dt;
    updateMat_ = cv::Mat::eye(ndim, 2 * ndim, CV_64F);
}

void KalmanFilterXYWH::initiate(const cv::Vec4f& measurement, cv::Mat& mean, cv::Mat& covariance) const {
    mean = cv::Mat::zeros(8, 1, CV_64F);
    for (int i = 0; i < 4; i++) mean.at<double>(i) = measurement[i];

    double w = measurement[2], h = measurement[3];
    double std_[8] = {
        2 * stdWeightPosition_ * w, 2 * stdWeightPosition_ * h,
        2 * stdWeightPosition_ * w, 2 * stdWeightPosition_ * h,
        10 * stdWeightVelocity_ * w, 10 * stdWeightVelocity_ * h,
        10 * stdWeightVelocity_ * w, 10 * stdWeightVelocity_ * h,
    };
    covariance = cv::Mat::zeros(8, 8, CV_64F);
    for (int i = 0; i < 8; i++) covariance.at<double>(i, i) = std_[i] * std_[i];
}

void KalmanFilterXYWH::predict(cv::Mat& mean, cv::Mat& covariance) const {
    double w = mean.at<double>(2), h = mean.at<double>(3);
    double stdPos[4] = {
        stdWeightPosition_ * w, stdWeightPosition_ * h,
        stdWeightPosition_ * w, stdWeightPosition_ * h,
    };
    double stdVel[4] = {
        stdWeightVelocity_ * w, stdWeightVelocity_ * h,
        stdWeightVelocity_ * w, stdWeightVelocity_ * h,
    };
    cv::Mat motionCov = cv::Mat::zeros(8, 8, CV_64F);
    for (int i = 0; i < 4; i++) {
        motionCov.at<double>(i, i) = stdPos[i] * stdPos[i];
        motionCov.at<double>(4 + i, 4 + i) = stdVel[i] * stdVel[i];
    }
    mean = motionMat_ * mean;
    covariance = motionMat_ * covariance * motionMat_.t() + motionCov;
}

void KalmanFilterXYWH::project(const cv::Mat& mean, const cv::Mat& covariance,
                                cv::Mat& projMean, cv::Mat& projCovariance) const {
    double w = mean.at<double>(2), h = mean.at<double>(3);
    double std_[4] = {
        stdWeightPosition_ * w, stdWeightPosition_ * h,
        stdWeightPosition_ * w, stdWeightPosition_ * h,
    };
    cv::Mat innovationCov = cv::Mat::zeros(4, 4, CV_64F);
    for (int i = 0; i < 4; i++) innovationCov.at<double>(i, i) = std_[i] * std_[i];

    projMean = updateMat_ * mean;
    projCovariance = updateMat_ * covariance * updateMat_.t() + innovationCov;
}

void KalmanFilterXYWH::update(cv::Mat& mean, cv::Mat& covariance, const cv::Vec4f& measurement) const {
    cv::Mat projMean, projCovariance;
    project(mean, covariance, projMean, projCovariance);

    // Kalman gain = covariance @ H^T @ inv(projCovariance), solved via cv::solve
    // (equivalent to, but more stable than, forming the inverse explicitly).
    cv::Mat covHt = covariance * updateMat_.t();       // 8x4
    cv::Mat gainT;                                      // solves projCovariance * gainT = covHt^T
    cv::solve(projCovariance, covHt.t(), gainT, cv::DECOMP_CHOLESKY);
    cv::Mat kalmanGain = gainT.t();                      // 8x4

    cv::Mat meas = cv::Mat(4, 1, CV_64F);
    for (int i = 0; i < 4; i++) meas.at<double>(i) = measurement[i];
    cv::Mat innovation = meas - projMean;                // 4x1

    mean = mean + kalmanGain * innovation;
    covariance = covariance - kalmanGain * projCovariance * kalmanGain.t();
}
