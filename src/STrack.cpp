#include "STrack.hpp"
#include <cmath>

namespace {
int g_nextId = 0;
}

int STrack::nextId() { return ++g_nextId; }

STrack::STrack(const cv::Rect2f& tlwh, float score, int cls, const cv::Mat& feat)
    : score(score), cls(cls), tlwh_(tlwh) {
    if (!feat.empty()) updateFeatures(feat);
}

void STrack::updateFeatures(const cv::Mat& feat) {
    cv::Mat f;
    double n = cv::norm(feat, cv::NORM_L2);
    f = (n > 1e-12) ? feat / n : feat;
    currFeat = f.clone();
    if (smoothFeat.empty()) {
        smoothFeat = f.clone();
    } else {
        smoothFeat = kFeatAlpha * smoothFeat + (1.0 - kFeatAlpha) * f;
    }
    double sn = cv::norm(smoothFeat, cv::NORM_L2);
    if (sn > 1e-12) smoothFeat /= sn;
}

cv::Rect2f STrack::tlwh() const {
    if (mean.empty()) return tlwh_;
    float x = (float)mean.at<double>(0);
    float y = (float)mean.at<double>(1);
    float w = (float)mean.at<double>(2);
    float h = (float)mean.at<double>(3);
    return cv::Rect2f(x - w / 2.0f, y - h / 2.0f, w, h);
}

cv::Rect2f STrack::tlbr() const {
    cv::Rect2f t = tlwh();
    return cv::Rect2f(t.x, t.y, t.x + t.width, t.y + t.height); // (x1,y1,x2,y2) packed in a Rect2f
}

void STrack::activate(const KalmanFilterXYWH& kf, int frameIdIn) {
    trackId = nextId();
    cv::Vec4f meas(tlwh_.x + tlwh_.width / 2.0f, tlwh_.y + tlwh_.height / 2.0f,
                   tlwh_.width, tlwh_.height);
    kf.initiate(meas, mean, covariance);
    tracklet_len = 0;
    state = TrackState::Tracked;
    if (frameIdIn == 1) isActivated = true;
    frameId = frameIdIn;
    startFrame = frameIdIn;
}

void STrack::reActivate(const STrack& newTrack, const KalmanFilterXYWH& kf, int frameIdIn, bool newId) {
    cv::Rect2f t = newTrack.tlwh();
    cv::Vec4f meas(t.x + t.width / 2.0f, t.y + t.height / 2.0f, t.width, t.height);
    kf.update(mean, covariance, meas);
    if (!newTrack.currFeat.empty()) updateFeatures(newTrack.currFeat);
    tracklet_len = 0;
    state = TrackState::Tracked;
    isActivated = true;
    frameId = frameIdIn;
    if (newId) trackId = nextId();
    score = newTrack.score;
    cls = newTrack.cls;
    idx = newTrack.idx;
}

void STrack::update(const STrack& newTrack, const KalmanFilterXYWH& kf, int frameIdIn) {
    frameId = frameIdIn;
    tracklet_len++;
    cv::Rect2f t = newTrack.tlwh();
    cv::Vec4f meas(t.x + t.width / 2.0f, t.y + t.height / 2.0f, t.width, t.height);
    kf.update(mean, covariance, meas);
    if (!newTrack.currFeat.empty()) updateFeatures(newTrack.currFeat);
    state = TrackState::Tracked;
    isActivated = true;
    score = newTrack.score;
    cls = newTrack.cls;
    idx = newTrack.idx;
}

void STrack::predict(const KalmanFilterXYWH& kf) {
    cv::Mat meanState = mean.clone();
    if (state != TrackState::Tracked) {
        meanState.at<double>(6) = 0.0; // vw
        meanState.at<double>(7) = 0.0; // vh
    }
    mean = meanState;
    kf.predict(mean, covariance);
}

void STrack::multiPredict(std::vector<STrack*>& tracks, const KalmanFilterXYWH& kf) {
    for (auto* t : tracks) t->predict(kf);
}

void STrack::multiGmc(std::vector<STrack*>& tracks, const cv::Mat& warp2x3) {
    if (tracks.empty()) return;
    // R8x8 = kron(I4, R): applies the affine's 2x2 rotation/scale block to each
    // (x,y)/(w,h)/(vx,vy)/(vw,vh) pair, translation added to position only —
    // matches STrack.multi_gmc exactly, including its w/h-rotation quirk.
    cv::Mat R = warp2x3(cv::Rect(0, 0, 2, 2));
    cv::Mat t = warp2x3(cv::Rect(2, 0, 1, 2));
    cv::Mat R64, t64;
    R.convertTo(R64, CV_64F);
    t.convertTo(t64, CV_64F);

    cv::Mat R8 = cv::Mat::zeros(8, 8, CV_64F);
    for (int b = 0; b < 4; b++)
        R64.copyTo(R8(cv::Rect(b * 2, b * 2, 2, 2)));

    for (auto* trk : tracks) {
        if (trk->mean.empty()) continue;
        cv::Mat newMean = R8 * trk->mean;
        newMean.at<double>(0) += t64.at<double>(0);
        newMean.at<double>(1) += t64.at<double>(1);
        trk->mean = newMean;
        trk->covariance = R8 * trk->covariance * R8.t();
    }
}
