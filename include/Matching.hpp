#pragma once
#include <vector>
#include "STrack.hpp"

// Port of ultralytics/trackers/utils/matching.py, used by BotSort.
namespace Matching {

// 1 - IoU(track.tlbr, detection.tlbr) cost matrix, size tracks.size() x dets.size().
std::vector<std::vector<double>> iouDistance(const std::vector<STrack*>& tracks,
                                              const std::vector<STrack*>& dets);

// Fuses detection confidence into an IoU cost matrix: fuse_sim = (1-cost) * det_score,
// returns 1 - fuse_sim. Mirrors matching.fuse_score.
void fuseScore(std::vector<std::vector<double>>& costMatrix, const std::vector<STrack*>& dets);

// Cosine-distance matrix between track.smoothFeat and det.currFeat (empty features -> 1.0).
std::vector<std::vector<double>> embeddingDistance(const std::vector<STrack*>& tracks,
                                                    const std::vector<STrack*>& dets);

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches; // (row, col)
    std::vector<int> unmatchedRows;
    std::vector<int> unmatchedCols;
};

// Solves the assignment via the Hungarian algorithm, then rejects any pair
// whose cost exceeds thresh (moving both sides to the unmatched lists) —
// equivalent in effect to lap.lapjv(..., cost_limit=thresh).
// nRows/nCols are passed explicitly (rather than inferred from costMatrix's
// shape) because a 0-row cost matrix is represented as an empty outer vector,
// which would otherwise silently lose the true detection count.
AssignmentResult linearAssignment(const std::vector<std::vector<double>>& costMatrix,
                                   int nRows, int nCols, double thresh);

} // namespace Matching
