#include "Hungarian.hpp"
#include <limits>

// Textbook O(rows^2 * cols) Kuhn-Munkres with potentials (1-indexed internally).
// Requires rows <= cols; the caller transposes when the input is the other way
// around so every "row" always finds a match.
std::vector<int> Hungarian::solve(const std::vector<std::vector<double>>& costIn) {
    int n = (int)costIn.size();
    if (n == 0) return {};
    int m = (int)costIn[0].size();
    if (m == 0) return std::vector<int>(n, -1);

    bool transposed = false;
    std::vector<std::vector<double>> a;
    int rows, cols;
    if (n <= m) {
        a = costIn;
        rows = n; cols = m;
    } else {
        transposed = true;
        a.assign(m, std::vector<double>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                a[j][i] = costIn[i][j];
        rows = m; cols = n;
    }

    const double INF = std::numeric_limits<double>::max() / 4;
    std::vector<double> u(rows + 1, 0.0), v(cols + 1, 0.0);
    std::vector<int> p(cols + 1, 0), way(cols + 1, 0);

    for (int i = 1; i <= rows; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(cols + 1, INF);
        std::vector<bool> used(cols + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            double delta = INF;
            for (int j = 1; j <= cols; j++) {
                if (!used[j]) {
                    double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= cols; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> rowToCol(rows, -1);
    for (int j = 1; j <= cols; j++)
        if (p[j] != 0) rowToCol[p[j] - 1] = j - 1;

    if (!transposed) {
        return rowToCol; // size n, values in [0, m)
    }
    std::vector<int> result(n, -1);
    for (int k = 0; k < rows; k++) { // k = original column index
        int origRow = rowToCol[k];
        if (origRow >= 0 && origRow < n) result[origRow] = k;
    }
    return result;
}
