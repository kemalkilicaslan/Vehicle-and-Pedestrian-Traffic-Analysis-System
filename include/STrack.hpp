#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include "KalmanFilterXYWH.hpp"

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

// Port of ultralytics/trackers/bot_sort.py: BOTrack (an STrack specialized to
// the xywh Kalman state, i.e. no aspect-ratio bookkeeping). One instance is
// used both as a raw "detection" (before activate()) and as a live track
// (after), exactly like the Python class.
class STrack {
public:
    STrack(const cv::Rect2f& tlwh, float score, int cls, const cv::Mat& feat = cv::Mat());

    void activate(const KalmanFilterXYWH& kf, int frameId);
    void reActivate(const STrack& newTrack, const KalmanFilterXYWH& kf, int frameId, bool newId = false);
    void update(const STrack& newTrack, const KalmanFilterXYWH& kf, int frameId);
    void predict(const KalmanFilterXYWH& kf);
    void markLost() { state = TrackState::Lost; }
    void markRemoved() { state = TrackState::Removed; }

    static void multiPredict(std::vector<STrack*>& tracks, const KalmanFilterXYWH& kf);
    // Applies a 2x3 affine warp (from GMC) to the mean/covariance of each track,
    // mirroring STrack.multi_gmc in the Python tracker.
    static void multiGmc(std::vector<STrack*>& tracks, const cv::Mat& warp2x3);

    cv::Rect2f tlwh() const;      // (x, y, w, h) top-left form
    cv::Rect2f tlbr() const;      // (x1, y1, x2, y2)

    void updateFeatures(const cv::Mat& feat);

    static int nextId();

    int trackId = 0;
    bool isActivated = false;
    TrackState state = TrackState::New;
    int frameId = 0;
    int startFrame = 0;
    int tracklet_len = 0;

    float score;
    int cls;
    int idx = -1; // index of the source detection in the current frame's list

    cv::Mat mean;        // 8x1, valid once activated
    cv::Mat covariance;  // 8x8, valid once activated

    cv::Mat currFeat;    // 1xD, latest ReID embedding (empty if ReID disabled)
    cv::Mat smoothFeat;  // 1xD, EMA-smoothed embedding
    static constexpr double kFeatAlpha = 0.9;

private:
    cv::Rect2f tlwh_; // raw detection box, used until the Kalman filter is initiated
};
