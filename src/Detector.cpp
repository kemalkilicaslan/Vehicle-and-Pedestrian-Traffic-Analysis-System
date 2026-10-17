#include "Detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

Detector::Detector(const std::string& modelPath, int imgSize, float confThreshold,
                    float iouThreshold, const std::set<int>& classFilter, const std::string& device)
    : imgSize_(imgSize), confThreshold_(confThreshold), iouThreshold_(iouThreshold),
      classFilter_(classFilter) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    if (net_.empty())
        throw std::runtime_error("[ERROR] Could not load ONNX model: " + modelPath);

    // "auto" tries CUDA, then OpenCL, then falls back to plain CPU.
    std::string dev = device;
    if (dev == "auto") {
#ifdef HAVE_OPENCV_DNN_CUDA
        dev = "cuda";
#else
        dev = cv::ocl::haveOpenCL() ? "opencl" : "cpu";
#endif
    }

    if (dev == "cuda") {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else if (dev == "opencl") {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    std::cout << "[INFO] Detector device: " << dev << "\n";
}

cv::Mat Detector::letterbox(const cv::Mat& frame, LetterboxInfo& info) const {
    int w = frame.cols, h = frame.rows;
    double gain = std::min((double)imgSize_ / w, (double)imgSize_ / h);
    int newW = (int)std::round(w * gain);
    int newH = (int)std::round(h * gain);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    double padW = (imgSize_ - newW) / 2.0;
    double padH = (imgSize_ - newH) / 2.0;
    int top = (int)std::round(padH - 0.1);
    int left = (int)std::round(padW - 0.1);
    int bottom = imgSize_ - newH - top;
    int right = imgSize_ - newW - left;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT,
                        cv::Scalar(114, 114, 114));
    info.gain = gain;
    info.padX = left;
    info.padY = top;
    return out;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    LetterboxInfo info;
    cv::Mat lb = letterbox(frame, info);

    cv::Mat blob = cv::dnn::blobFromImage(lb, 1.0 / 255.0, cv::Size(imgSize_, imgSize_),
                                           cv::Scalar(0, 0, 0), /*swapRB=*/true, /*crop=*/false, CV_32F);
    net_.setInput(blob);
    cv::Mat output = net_.forward(); // [1, 4+numClasses, numBoxes]

    int C = output.size[1];
    int N = output.size[2];
    numClasses_ = C - 4;
    const float* data = output.ptr<float>(0);

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    boxes.reserve(N);
    scores.reserve(N);
    classIds.reserve(N);

    for (int n = 0; n < N; n++) {
        float cx = data[0 * N + n];
        float cy = data[1 * N + n];
        float bw = data[2 * N + n];
        float bh = data[3 * N + n];

        float bestScore = -1.0f;
        int bestCls = -1;
        for (int c = 0; c < numClasses_; c++) {
            float s = data[(4 + c) * N + n];
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestScore <= confThreshold_) continue;
        if (!classFilter_.empty() && !classFilter_.count(bestCls)) continue;

        double x1 = (cx - bw / 2.0 - info.padX) / info.gain;
        double y1 = (cy - bh / 2.0 - info.padY) / info.gain;
        double x2 = (cx + bw / 2.0 - info.padX) / info.gain;
        double y2 = (cy + bh / 2.0 - info.padY) / info.gain;
        x1 = std::clamp(x1, 0.0, (double)frame.cols - 1);
        y1 = std::clamp(y1, 0.0, (double)frame.rows - 1);
        x2 = std::clamp(x2, 0.0, (double)frame.cols - 1);
        y2 = std::clamp(y2, 0.0, (double)frame.rows - 1);
        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back(cv::Point((int)std::round(x1), (int)std::round(y1)),
                            cv::Point((int)std::round(x2), (int)std::round(y2)));
        scores.push_back(bestScore);
        classIds.push_back(bestCls);
    }

    // Per-class NMS, matching ultralytics' default (non class-agnostic) NMS.
    std::vector<Detection> results;
    std::set<int> uniqueClasses(classIds.begin(), classIds.end());
    for (int cls : uniqueClasses) {
        std::vector<cv::Rect> clsBoxes;
        std::vector<float> clsScores;
        std::vector<int> origIdx;
        for (size_t i = 0; i < boxes.size(); i++) {
            if (classIds[i] == cls) {
                clsBoxes.push_back(boxes[i]);
                clsScores.push_back(scores[i]);
                origIdx.push_back((int)i);
            }
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(clsBoxes, clsScores, confThreshold_, iouThreshold_, keep);
        for (int k : keep) {
            int i = origIdx[k];
            Detection d;
            d.tlwh = cv::Rect2f((float)boxes[i].x, (float)boxes[i].y,
                                 (float)boxes[i].width, (float)boxes[i].height);
            d.score = scores[i];
            d.cls = classIds[i];
            results.push_back(d);
        }
    }
    return results;
}
