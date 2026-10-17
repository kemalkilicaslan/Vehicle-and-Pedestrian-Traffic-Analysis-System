#include "Matching.hpp"
#include "Hungarian.hpp"
#include <algorithm>

namespace {

double iouOf(const cv::Rect2f& a, const cv::Rect2f& b) {
    // a, b store (x1, y1, x2, y2) packed into a Rect2f (as produced by STrack::tlbr()).
    float ix1 = std::max(a.x, b.x);
    float iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.width, b.width);
    float iy2 = std::min(a.height, b.height);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = std::max(0.0f, a.width - a.x) * std::max(0.0f, a.height - a.y);
    float areaB = std::max(0.0f, b.width - b.x) * std::max(0.0f, b.height - b.y);
    float uni = areaA + areaB - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

} // namespace

namespace Matching {

std::vector<std::vector<double>> iouDistance(const std::vector<STrack*>& tracks,
                                              const std::vector<STrack*>& dets) {
    std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(dets.size(), 1.0));
    for (size_t i = 0; i < tracks.size(); i++) {
        cv::Rect2f a = tracks[i]->tlbr();
        for (size_t j = 0; j < dets.size(); j++) {
            cv::Rect2f b = dets[j]->tlbr();
            cost[i][j] = 1.0 - iouOf(a, b);
        }
    }
    return cost;
}

void fuseScore(std::vector<std::vector<double>>& costMatrix, const std::vector<STrack*>& dets) {
    if (costMatrix.empty()) return;
    for (auto& row : costMatrix) {
        for (size_t j = 0; j < row.size(); j++) {
            double iouSim = 1.0 - row[j];
            double fuseSim = iouSim * dets[j]->score;
            row[j] = 1.0 - fuseSim;
        }
    }
}

std::vector<std::vector<double>> embeddingDistance(const std::vector<STrack*>& tracks,
                                                    const std::vector<STrack*>& dets) {
    std::vector<std::vector<double>> cost(tracks.size(), std::vector<double>(dets.size(), 1.0));
    for (size_t i = 0; i < tracks.size(); i++) {
        if (tracks[i]->smoothFeat.empty()) continue;
        for (size_t j = 0; j < dets.size(); j++) {
            if (dets[j]->currFeat.empty()) continue;
            double dot = tracks[i]->smoothFeat.dot(dets[j]->currFeat);
            double d = 1.0 - dot; // cosine distance (features are L2-normalized)
            cost[i][j] = std::max(0.0, d);
        }
    }
    return cost;
}

AssignmentResult linearAssignment(const std::vector<std::vector<double>>& costMatrix,
                                   int nRows, int nCols, double thresh) {
    AssignmentResult result;
    if (nRows == 0 || nCols == 0) {
        for (int i = 0; i < nRows; i++) result.unmatchedRows.push_back(i);
        for (int j = 0; j < nCols; j++) result.unmatchedCols.push_back(j);
        return result;
    }

    std::vector<int> rowToCol = Hungarian::solve(costMatrix);
    std::vector<bool> colUsed(nCols, false);
    for (int i = 0; i < nRows; i++) {
        int j = rowToCol[i];
        if (j >= 0 && costMatrix[i][j] <= thresh) {
            result.matches.emplace_back(i, j);
            colUsed[j] = true;
        } else {
            result.unmatchedRows.push_back(i);
        }
    }
    for (int j = 0; j < nCols; j++)
        if (!colUsed[j]) result.unmatchedCols.push_back(j);
    return result;
}

} // namespace Matching
