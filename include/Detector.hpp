#pragma once
#include <opencv2/dnn.hpp>
#include <vector>
#include <set>
#include "BotSort.hpp"

// Runs the Ultralytics-exported visdrone.onnx detect head (verified layout:
// input "images" [1,3,imgSize,imgSize], output [1, 4+numClasses, numBoxes],
// box in letterboxed-image xywh pixels, class scores already sigmoid'd, no
// built-in NMS) through OpenCV's DNN module and turns it into plain
// Detection boxes in original-frame coordinates, post class-wise NMS.
class Detector {
public:
    Detector(const std::string& modelPath, int imgSize, float confThreshold,
             float iouThreshold, const std::set<int>& classFilter, const std::string& device);

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    struct LetterboxInfo {
        double gain;
        double padX;
        double padY;
    };

    cv::Mat letterbox(const cv::Mat& frame, LetterboxInfo& info) const;

    cv::dnn::Net net_;
    int imgSize_;
    float confThreshold_;
    float iouThreshold_;
    std::set<int> classFilter_;
    int numClasses_ = 0;
};
