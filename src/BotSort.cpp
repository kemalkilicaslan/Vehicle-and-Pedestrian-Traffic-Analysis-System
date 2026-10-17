#include "BotSort.hpp"
#include "Matching.hpp"
#include <algorithm>
#include <set>

namespace {

std::vector<STrack*> toRaw(const std::vector<std::shared_ptr<STrack>>& v) {
    std::vector<STrack*> out;
    out.reserve(v.size());
    for (auto& p : v) out.push_back(p.get());
    return out;
}

// Combine two track lists, de-duplicated by track_id, `a` entries win ties
// and keep their position — mirrors byte_tracker.joint_stracks.
std::vector<std::shared_ptr<STrack>> jointStracks(const std::vector<std::shared_ptr<STrack>>& a,
                                                   const std::vector<std::shared_ptr<STrack>>& b) {
    std::vector<std::shared_ptr<STrack>> out = a;
    std::set<int> seen;
    for (auto& t : a) seen.insert(t->trackId);
    for (auto& t : b) {
        if (seen.insert(t->trackId).second) out.push_back(t);
    }
    return out;
}

// a minus tracks whose track_id appears in b — mirrors byte_tracker.sub_stracks.
std::vector<std::shared_ptr<STrack>> subStracks(const std::vector<std::shared_ptr<STrack>>& a,
                                                 const std::vector<std::shared_ptr<STrack>>& b) {
    std::set<int> remove;
    for (auto& t : b) remove.insert(t->trackId);
    std::vector<std::shared_ptr<STrack>> out;
    for (auto& t : a)
        if (!remove.count(t->trackId)) out.push_back(t);
    return out;
}

} // namespace

BotSort::BotSort(const BotSortParams& params) : params_(params), gmc_(2) {
    maxTimeLost_ = std::max(1, (int)(params_.frameRate / 30.0 * params_.trackBuffer));
}

std::vector<std::shared_ptr<STrack>> BotSort::makeTracks(const std::vector<Detection>& dets,
                                                          const cv::Mat& frame) const {
    std::vector<std::shared_ptr<STrack>> out;
    out.reserve(dets.size());
    for (auto& d : dets) {
        cv::Mat feat;
        if (params_.withReid && reidEncoder_) feat = reidEncoder_(frame, d.tlwh);
        out.push_back(std::make_shared<STrack>(d.tlwh, d.score, d.cls, feat));
    }
    return out;
}

std::vector<std::vector<double>> BotSort::getDists(const std::vector<STrack*>& tracks,
                                                    const std::vector<STrack*>& dets) const {
    auto dists = Matching::iouDistance(tracks, dets);
    std::vector<std::vector<bool>> mask(dists.size());
    for (size_t i = 0; i < dists.size(); i++) {
        mask[i].resize(dists[i].size());
        for (size_t j = 0; j < dists[i].size(); j++)
            mask[i][j] = dists[i][j] > params_.proximityThresh;
    }
    if (params_.fuseScore) Matching::fuseScore(dists, dets);
    if (params_.withReid) {
        auto emb = Matching::embeddingDistance(tracks, dets);
        for (size_t i = 0; i < emb.size(); i++) {
            for (size_t j = 0; j < emb[i].size(); j++) {
                double e = emb[i][j] / 2.0;
                if (e > params_.appearanceThresh) e = 1.0;
                if (mask[i][j]) e = 1.0;
                dists[i][j] = std::min(dists[i][j], e);
            }
        }
    }
    return dists;
}

std::vector<std::shared_ptr<STrack>> BotSort::update(const std::vector<Detection>& detections,
                                                      const cv::Mat& frame) {
    frameId_++;
    std::vector<std::shared_ptr<STrack>> activatedStracks, refindStracks, lostStracksThisFrame, removedStracksThisFrame;

    std::vector<Detection> highDets, lowDets;
    for (auto& d : detections) {
        if (d.score >= params_.trackHighThresh) highDets.push_back(d);
        else if (d.score > params_.trackLowThresh) lowDets.push_back(d);
    }
    std::vector<std::shared_ptr<STrack>> detsHigh = makeTracks(highDets, frame);
    std::vector<std::shared_ptr<STrack>> detsLow = makeTracks(lowDets, frame);

    std::vector<std::shared_ptr<STrack>> unconfirmed, trackedConfirmed;
    for (auto& t : trackedStracks_) {
        if (!t->isActivated) unconfirmed.push_back(t);
        else trackedConfirmed.push_back(t);
    }

    std::vector<std::shared_ptr<STrack>> strackPool = jointStracks(trackedConfirmed, lostStracks_);

    auto poolPtrs = toRaw(strackPool);
    STrack::multiPredict(poolPtrs, kalmanFilter_);

    cv::Mat warp = gmc_.apply(frame);
    STrack::multiGmc(poolPtrs, warp);
    auto unconfirmedPtrs = toRaw(unconfirmed);
    STrack::multiGmc(unconfirmedPtrs, warp);

    // --- First association: high-score detections vs. the full pool ---
    auto detsHighPtrs = toRaw(detsHigh);
    auto dists = getDists(poolPtrs, detsHighPtrs);
    auto assign = Matching::linearAssignment(dists, (int)poolPtrs.size(), (int)detsHighPtrs.size(), params_.matchThresh);

    for (auto& [ti, di] : assign.matches) {
        auto& track = strackPool[ti];
        auto& det = detsHigh[di];
        if (track->state == TrackState::Tracked) {
            track->update(*det, kalmanFilter_, frameId_);
            activatedStracks.push_back(track);
        } else {
            track->reActivate(*det, kalmanFilter_, frameId_, false);
            refindStracks.push_back(track);
        }
    }

    // --- Second association: low-score detections vs. still-tracked leftovers ---
    std::vector<std::shared_ptr<STrack>> rTrackedStracks;
    for (int idx : assign.unmatchedRows)
        if (strackPool[idx]->state == TrackState::Tracked) rTrackedStracks.push_back(strackPool[idx]);

    auto rTrackedPtrs = toRaw(rTrackedStracks);
    auto detsLowPtrs = toRaw(detsLow);
    auto dists2 = Matching::iouDistance(rTrackedPtrs, detsLowPtrs);
    auto assign2 = Matching::linearAssignment(dists2, (int)rTrackedPtrs.size(), (int)detsLowPtrs.size(), 0.5);

    for (auto& [ti, di] : assign2.matches) {
        auto& track = rTrackedStracks[ti];
        auto& det = detsLow[di];
        if (track->state == TrackState::Tracked) {
            track->update(*det, kalmanFilter_, frameId_);
            activatedStracks.push_back(track);
        } else {
            track->reActivate(*det, kalmanFilter_, frameId_, false);
            refindStracks.push_back(track);
        }
    }
    for (int idx : assign2.unmatchedRows) {
        auto& track = rTrackedStracks[idx];
        if (track->state != TrackState::Lost) {
            track->markLost();
            lostStracksThisFrame.push_back(track);
        }
    }

    // --- Unconfirmed tracks (single-frame tracklets) vs. remaining high-score detections ---
    std::vector<std::shared_ptr<STrack>> detsRemaining;
    for (int idx : assign.unmatchedCols) detsRemaining.push_back(detsHigh[idx]);

    auto detsRemainingPtrs = toRaw(detsRemaining);
    auto dists3 = getDists(unconfirmedPtrs, detsRemainingPtrs);
    auto assign3 = Matching::linearAssignment(dists3, (int)unconfirmedPtrs.size(), (int)detsRemainingPtrs.size(), 0.7);

    for (auto& [ti, di] : assign3.matches) {
        unconfirmed[ti]->update(*detsRemaining[di], kalmanFilter_, frameId_);
        activatedStracks.push_back(unconfirmed[ti]);
    }
    for (int idx : assign3.unmatchedRows) {
        auto& track = unconfirmed[idx];
        track->markRemoved();
        removedStracksThisFrame.push_back(track);
    }

    // --- Spawn new tracks from whatever high-score detections are still unmatched ---
    for (int idx : assign3.unmatchedCols) {
        auto& track = detsRemaining[idx];
        if (track->score < params_.newTrackThresh) continue;
        track->activate(kalmanFilter_, frameId_);
        activatedStracks.push_back(track);
    }

    // --- Timeout lost tracks that have been missing for too long ---
    std::vector<std::shared_ptr<STrack>> stillLost;
    for (auto& t : lostStracks_) {
        if (frameId_ - t->frameId > maxTimeLost_) {
            t->markRemoved();
            removedStracksThisFrame.push_back(t);
        } else {
            stillLost.push_back(t);
        }
    }
    lostStracks_ = stillLost;

    // --- Rebuild the persistent lists for next frame ---
    std::vector<std::shared_ptr<STrack>> newTracked;
    for (auto& t : trackedStracks_)
        if (t->state == TrackState::Tracked) newTracked.push_back(t);
    newTracked = jointStracks(newTracked, activatedStracks);
    newTracked = jointStracks(newTracked, refindStracks);
    trackedStracks_ = newTracked;

    lostStracks_ = subStracks(lostStracks_, trackedStracks_);
    for (auto& t : lostStracksThisFrame) lostStracks_.push_back(t);

    removedStracks_.insert(removedStracks_.end(), removedStracksThisFrame.begin(), removedStracksThisFrame.end());
    if (removedStracks_.size() > 1000)
        removedStracks_.erase(removedStracks_.begin(), removedStracks_.end() - 999);

    std::vector<std::shared_ptr<STrack>> result;
    for (auto& t : trackedStracks_)
        if (t->isActivated) result.push_back(t);
    return result;
}
