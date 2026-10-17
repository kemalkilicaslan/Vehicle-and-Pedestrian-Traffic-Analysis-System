#pragma once
#include <vector>

// Classic O(n^3) Hungarian (Kuhn-Munkres) minimum-cost assignment on a
// rectangular cost matrix. Non-square input is padded internally with a
// large sentinel cost so padded (fake) rows/cols never win a real match.
// Returns, for each real row, the assigned real column index or -1 if the
// matrix was empty on that side.
class Hungarian {
public:
    static std::vector<int> solve(const std::vector<std::vector<double>>& cost);
};
