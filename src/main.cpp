// Vehicle and Pedestrian Traffic Analysis System — C++ port of
// Vehicle-and-Pedestrian-Traffic-Analysis-System.py, using visdrone.onnx (via
// OpenCV's DNN module) for detection and a from-scratch BoT-SORT
// (Kalman filter + GMC + cascaded IoU/ReID matching) in place of
// ultralytics' model.track(tracker="botsort.yaml").

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "BotSort.hpp"
#include "Detector.hpp"

namespace fs = std::filesystem;

// ----------------------------- Configuration -----------------------------
static const std::string VIDEO_PATH = "Vehicle-and-Pedestrian-Traffic.mp4";
static const std::string MODEL_PATH = "visdrone.onnx"; // Ultralytics YOLO26x, 50 epochs on VisDrone, mAP50 61.8% (ONNX export of visdrone.pt, for OpenCV's DNN module)
static const float CONF_THRESHOLD = 0.0f;   // low value feeds BoT-SORT's low-score matching, a new track opens above ~0.25
static const float IOU_THRESHOLD = 0.3f;    // NMS IoU threshold; recommended for crowding with a fine-tuned model
static const int IMG_SIZE = 1280;           // the model was trained at imgsz=1280, keeping the same value is recommended
static const int STRIDE = 1;                // process every Nth frame (>1 speeds up but weakens ID continuity)
static const double MOTION_THRESH = 1.5;    // pixel displacement over the motion window above which an object is moving
static const size_t MOTION_WINDOW = 5;      // number of recent processed frames used to judge motion (stationary = green)
static const int MIN_MOVING_PEDESTRIANS = 3; // crosswalk mask triggers once this many moving pedestrians are inside it
static const int PEDESTRIAN_OVERRIDE_MARGIN = 5; // crosswalk mask forces blue (never red) once pedestrian count
                                                  // exceeds motorized vehicle count by at least this many, in that zone
static const std::string DEVICE = "auto";   // auto | cpu | cuda | opencl (OpenCV DNN has no MPS backend; "auto" tries CUDA, then OpenCL, then CPU)
static const int MAX_FRAMES = 0;            // maximum frames to process (0 = whole video), for quick tests
static const bool SHOW_PREVIEW = true;      // live preview; falls back to headless recording if no GUI is available
static const bool CHECK_ROI = false;        // validate ROIs on the first frame and exit, instead of running the analysis

// Live preview window is scaled down to fit within this box (output video file is unaffected —
// it's always written at full W x outH); without this, a tall dashboard + 1080p video can exceed
// the screen's usable height and the window manager pushes its top (the panel) off-screen.
static const int MAX_PREVIEW_WIDTH = 1280;
static const int MAX_PREVIEW_HEIGHT = 800;

// Reference resolution used to scale the ROI polygons for non-standard video sizes
static const int REF_W = 1920, REF_H = 1080;

using Polygon = std::vector<cv::Point>;
using ROIMap = std::vector<std::pair<std::string, Polygon>>;

// ROI polygon coordinates (reference resolution 1920x1080). Order matters:
// find_zone() returns the first match, so crossings are listed before the
// generic road/sidewalk zones (no double counting at shared borders).
static ROIMap makeRoi() {
    return {
        {"CROSSWALK_S", {  // roi_1 - zebra-striped crosswalk
            {220, 713}, {1528, 625}, {1517, 759}, {1532, 916}, {20, 1070}, {180, 897}, {236, 798},
        }},
        {"CROSSWALK_NW", {  // roi_2 - zebra-striped crosswalk
            {212, 704}, {1043, 257}, {945, 193}, {132, 560},
        }},
        {"CROSSWALK_NE", {  // roi_3 - zebra-striped crosswalk
            {1681, 410}, {1751, 323}, {1360, 168}, {1252, 228},
        }},
        {"ROAD_MID", {  // roi_7 - open intersection surface, no crosswalk markings
            {222, 708}, {1529, 618}, {1576, 479}, {1670, 414}, {1251, 232}, {1053, 267},
        }},
        {"ROAD_SW", {  // roi_8 - paved road surface, no crosswalk markings
            {45, 1080}, {1531, 927}, {1607, 1080},
        }},
        {"ROAD_E", {  // roi_9 - road surface with vehicles
            {1373, 166}, {1752, 313}, {1920, 73}, {1920, 0}, {1597, 0},
        }},
        {"ROAD_NW", {  // roi_10 - roundabout/road surface with vehicles
            {132, 551}, {916, 192}, {939, 116}, {782, 0}, {0, 0}, {0, 343},
        }},
        {"SIDEWALK_E", {  // roi_4 - pedestrian-only area, no road markings
            {1620, 1080}, {1583, 1011}, {1548, 950}, {1533, 909}, {1530, 842}, {1522, 759},
            {1531, 633}, {1565, 537}, {1589, 473}, {1682, 411}, {1752, 323}, {1920, 85}, {1920, 1080},
        }},
        {"SIDEWALK_W", {  // roi_5 - pedestrian-only area, no road markings
            {0, 1080}, {96, 992}, {149, 925}, {188, 868}, {222, 807}, {227, 773},
            {215, 710}, {119, 536}, {0, 346},
        }},
        {"SIDEWALK_N", {  // roi_6 - pedestrian-only area, no road markings
            {791, 0}, {945, 114}, {922, 189}, {943, 188}, {1049, 257}, {1177, 244}, {1255, 225}, {1590, 0},
        }},
    };
}

// Crossings first: find_zone() returns the first match, so border points go to the crossing (no double count)
static const std::unordered_map<std::string, cv::Scalar> ZONE_COLORS = {  // BGR
    {"CROSSWALK_S",   cv::Scalar(255, 255, 0)},    // cyan
    {"CROSSWALK_NW",  cv::Scalar(255, 0, 255)},    // magenta
    {"CROSSWALK_NE",  cv::Scalar(0, 220, 0)},      // dark green
    {"ROAD_MID",      cv::Scalar(0, 255, 0)},      // green
    {"ROAD_SW",       cv::Scalar(200, 0, 200)},    // purple-magenta
    {"ROAD_E",        cv::Scalar(0, 150, 255)},    // orange
    {"ROAD_NW",       cv::Scalar(0, 230, 255)},    // yellow
    {"SIDEWALK_E",    cv::Scalar(0, 120, 255)},    // orange-2
    {"SIDEWALK_W",    cv::Scalar(200, 0, 120)},    // purple
    {"SIDEWALK_N",    cv::Scalar(0, 210, 235)},    // yellow-2
};

// VisDrone-DET class indices (visdrone.onnx, NOT COCO)
static const std::map<int, std::string> PERSON_CLASSES = {
    {0, "pedestrian"},
    {1, "people"},
};

static const std::map<int, std::string> VEHICLE_CLASSES = {
    {2, "bicycle"},
    {3, "car"},
    {4, "van"},
    {5, "truck"},
    {6, "tricycle"},
    {7, "awning-tricycle"},
    {8, "bus"},
    {9, "motor"},
};

static std::set<int> trackClasses() {
    std::set<int> s;
    for (auto& [k, v] : PERSON_CLASSES) s.insert(k);
    for (auto& [k, v] : VEHICLE_CLASSES) s.insert(k);
    return s;
}

// VEHICLE_CLASSES values() in class-id order (2..9) — used to order the ROAD pie's slices/legend
// and the panel's top legend, exactly like Python's VEHICLE_TYPE_ORDER.
static std::vector<std::string> vehicleTypeOrder() {
    std::vector<std::string> order;
    for (auto& [k, v] : VEHICLE_CLASSES) order.push_back(v);
    return order;
}
static const std::vector<std::string> VEHICLE_TYPE_ORDER = vehicleTypeOrder();

// Motorized vehicle types only — bicycle (human-powered) is the sole exclusion. Used to decide
// which vehicles can trigger the crosswalk red mask on a box/polygon border overlap.
static const std::set<std::string> MOTORIZED_VEHICLE_TYPES = {
    "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor",
};

static const std::unordered_map<std::string, cv::Scalar> VEHICLE_TYPE_COLORS = { // BGR, one distinct
    {"bicycle",         cv::Scalar(255, 255, 0)},   // cyan       color per vehicle type — used by
    {"car",             cv::Scalar(60, 160, 255)},  // orange     the panel's top legend, the ROAD
    {"van",             cv::Scalar(255, 200, 0)},   // blue       pie and its legend
    {"truck",           cv::Scalar(0, 100, 25)},    // green
    {"tricycle",        cv::Scalar(180, 90, 200)},  // purple
    {"awning-tricycle", cv::Scalar(140, 140, 140)}, // gray
    {"bus",             cv::Scalar(0, 210, 255)},   // gold
    {"motor",           cv::Scalar(220, 120, 255)}, // pink
};

// Visualization parameters
static const cv::Scalar MOVING_COLOR(0, 255, 0);         // green (BGR) — displaced beyond the motion threshold
static const cv::Scalar STATIONARY_COLOR(0, 0, 255);     // red (BGR)   — below the motion threshold
static const cv::Scalar CROSSWALK_PEDESTRIAN_COLOR(255, 0, 0); // blue (BGR) — fill when MIN_MOVING_PEDESTRIANS+ moving pedestrians are inside
static const cv::Scalar CROSSWALK_VEHICLE_COLOR(0, 0, 255);    // red (BGR)  — fill when any vehicle is inside (takes priority over pedestrians)
static const double CROSSWALK_MASK_ALPHA = 0.25;               // opacity of the crosswalk fill
static const int FONT = cv::FONT_HERSHEY_SIMPLEX;
static const bool SHOW_BOXES = false; // draw each tracked object's moving/stationary box

// ----------------------------- Helper functions -----------------------------

// Scales the ROIs proportionally if the video differs from the reference resolution.
static ROIMap scaleRois(const ROIMap& roi, int w, int h) {
    if (w == REF_W && h == REF_H) return roi;
    double sx = (double)w / REF_W, sy = (double)h / REF_H;
    std::cerr << "[WARNING] Video " << w << "x" << h << "; ROI coordinates scaled from the "
              << REF_W << "x" << REF_H << " reference (sx=" << sx << ", sy=" << sy << ").\n";
    ROIMap out;
    out.reserve(roi.size());
    for (auto& [name, poly] : roi) {
        Polygon scaled;
        scaled.reserve(poly.size());
        for (auto& p : poly)
            scaled.emplace_back((int)(p.x * sx), (int)(p.y * sy));
        out.emplace_back(name, scaled);
    }
    return out;
}

// Returns the first zone that contains the point (x, y), else nullopt.
static std::optional<std::string> findZone(const cv::Point2f& point, const ROIMap& roi) {
    for (auto& [name, poly] : roi) {
        if (cv::pointPolygonTest(poly, point, false) >= 0) return name;
    }
    return std::nullopt;
}

// Classifies an object as moving from the averaged displacement over its recent track history.
static bool isMoving(int trackId, const cv::Point2f& center,
                      std::unordered_map<int, std::deque<cv::Point2f>>& history, double threshold) {
    auto& hist = history[trackId];
    hist.push_back(center);
    if (hist.size() > MOTION_WINDOW) hist.pop_front();
    if (hist.size() < 2) return false;

    size_t half = std::max<size_t>(1, hist.size() / 2);
    cv::Point2f oldSum(0, 0), newSum(0, 0);
    for (size_t i = 0; i < half; i++) oldSum += hist[i];
    for (size_t i = hist.size() - half; i < hist.size(); i++) newSum += hist[i];
    cv::Point2f oldMean = oldSum / (float)half, newMean = newSum / (float)half;
    return cv::norm(newMean - oldMean) >= threshold;
}

// Tests whether segments (p1-p2) and (p3-p4) intersect (touching counts), via the standard
// orientation/on-segment test — used by boxIntersectsPolygon to detect a box edge crossing a
// polygon edge.
static bool segmentsIntersect(const cv::Point2f& p1, const cv::Point2f& p2,
                               const cv::Point2f& p3, const cv::Point2f& p4) {
    auto orientation = [](const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) -> int {
        double val = (double)(b.x - a.x) * (c.y - a.y) - (double)(b.y - a.y) * (c.x - a.x);
        if (val == 0) return 0;
        return val > 0 ? 1 : 2;
    };
    auto onSegment = [](const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) -> bool {
        return (std::min(a.x, c.x) <= b.x && b.x <= std::max(a.x, c.x) &&
                std::min(a.y, c.y) <= b.y && b.y <= std::max(a.y, c.y));
    };
    int o1 = orientation(p1, p2, p3), o2 = orientation(p1, p2, p4);
    int o3 = orientation(p3, p4, p1), o4 = orientation(p3, p4, p2);
    if (o1 != o2 && o3 != o4) return true;
    if (o1 == 0 && onSegment(p1, p3, p2)) return true;
    if (o2 == 0 && onSegment(p1, p4, p2)) return true;
    if (o3 == 0 && onSegment(p3, p1, p4)) return true;
    if (o4 == 0 && onSegment(p3, p2, p4)) return true;
    return false;
}

// Tests whether an axis-aligned detection box (x1, y1)-(x2, y2) intersects a polygon in any way:
// a box edge crosses a polygon edge, a polygon vertex falls inside the box, or a box corner falls
// inside the polygon (the last two cover one shape fully containing the other, where no edges
// actually cross). Used to trigger the crosswalk red mask on any border overlap, not just when
// the box's ground-contact point falls inside the zone.
static bool boxIntersectsPolygon(float x1, float y1, float x2, float y2, const Polygon& poly) {
    cv::Point2f corners[4] = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
    for (int i = 0; i < 4; i++) {
        cv::Point2f bp1 = corners[i], bp2 = corners[(i + 1) % 4];
        for (size_t j = 0; j < poly.size(); j++) {
            cv::Point2f pp1 = poly[j], pp2 = poly[(j + 1) % poly.size()];
            if (segmentsIntersect(bp1, bp2, pp1, pp2)) return true;
        }
    }
    for (auto& p : poly)
        if (x1 <= p.x && p.x <= x2 && y1 <= p.y && p.y <= y2) return true;
    for (auto& c : corners)
        if (cv::pointPolygonTest(poly, c, false) >= 0) return true;
    return false;
}

// Draws semi-transparent zone fills, borders and zone labels.
static cv::Mat drawRois(const cv::Mat& frameIn, const ROIMap& roi, double alpha = 0.16) {
    cv::Mat frame = frameIn;
    cv::Mat overlay = frame.clone();
    for (auto& [name, poly] : roi)
        cv::fillPoly(overlay, std::vector<Polygon>{poly}, ZONE_COLORS.at(name));
    cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);
    for (auto& [name, poly] : roi) {
        cv::polylines(frame, std::vector<Polygon>{poly}, true, ZONE_COLORS.at(name), 2, cv::LINE_AA);
        cv::Point2d mean(0, 0);
        for (auto& p : poly) mean += cv::Point2d(p.x, p.y);
        mean *= (1.0 / poly.size());
        cv::Point textOrg((int)mean.x - 60, (int)mean.y);
        cv::putText(frame, name, textOrg, FONT, 0.55, cv::Scalar(0, 0, 0), 0, cv::LINE_AA);
        cv::putText(frame, name, textOrg, FONT, 0.55, ZONE_COLORS.at(name), 1, cv::LINE_AA);
    }
    return frame;
}

// Fills the given crosswalk polygons with a semi-transparent color mask, one color per zone (see
// zoneColors: name -> BGR color) — red for a crosswalk with a vehicle inside, blue for one with
// MIN_MOVING_PEDESTRIANS+ moving pedestrians inside.
static cv::Mat drawCrosswalkMasks(const cv::Mat& frame, const ROIMap& roi,
                                   const std::unordered_map<std::string, cv::Scalar>& zoneColors,
                                   double alpha = CROSSWALK_MASK_ALPHA) {
    if (zoneColors.empty()) return frame;
    cv::Mat overlay = frame.clone();
    for (auto& [name, color] : zoneColors) {
        for (auto& [n, poly] : roi) {
            if (n == name) { cv::fillPoly(overlay, std::vector<Polygon>{poly}, color); break; }
        }
    }
    cv::Mat out;
    cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, out);
    return out;
}

// Zone categories shown as the panel's 3 columns, left to right; each ROI name must start
// with exactly one of these prefixes (enforced implicitly by ROI naming: CATEGORY_SUFFIX).
// Columns are spread across the full frame width (see drawPanel's colWidth), not a fixed pixel
// size, so the panel fills the frame instead of clustering in a corner.
static const std::vector<std::string> PANEL_CATEGORIES = {"CROSSWALK", "SIDEWALK", "ROAD"};
static const int PANEL_MARGIN = 24;                      // left/right margin of the whole panel
static const int PANEL_ROW_H = 46;                       // px, tall enough for the larger row font + bars
static const cv::Scalar PANEL_TEXT_COLOR(255, 255, 255); // all panel text is white

// Font scales — sized for a 1080p+ frame so the panel reads comfortably at a glance
static const double PANEL_TITLE_SCALE = 0.9;
static const double PANEL_LEGEND_SCALE = 0.65;
static const double PANEL_HEADER_SCALE = 0.85;
static const double PANEL_ZONE_LEGEND_SCALE = 0.6;
static const double PANEL_CAPTION_SCALE = 0.6;

static const std::string PANEL_TITLE = "VEHICLE AND PEDESTRIAN TRAFFIC ANALYSIS";

// Vertical layout offsets from the panel's top, shared by drawPanel() and panelHeight() so the
// two stay in sync: dashboard title -> [gap] -> one single legend line (every vehicle type, the
// colors used by the ROAD pie) -> [gap] -> category header -> [gap, sized for the large pie] ->
// pie row -> [gap] -> CROSSWALK/SIDEWALK's caption below their pie.
static const int PANEL_TITLE_Y = 34;
static const int PANEL_Y0 = PANEL_TITLE_Y + 32;
static const int PANEL_HEADER_Y = PANEL_Y0 + 78;
static const int PANEL_GRID_Y = PANEL_HEADER_Y + 130;  // clears the ROAD pie's 8-line vehicle-type legend above it
static const int PANEL_BOTTOM_PAD = 56;  // margin below the last row, sized to fit the caption below ROAD's 8-line legend

static const cv::Scalar PIE_EMPTY_COLOR(70, 70, 70);   // gray "no data" fill, for a pie with nothing to show
static const cv::Scalar PIE_OUTLINE_COLOR(20, 20, 20); // thin dark border so slices stay visually separated
static const int PIE_GAP_AFTER = 50;                   // px, gap between a pie and the text/legend next to it

// Every column renders exactly one pooled pie row (see drawPanel): CROSSWALK/SIDEWALK split their
// current (this-frame) pedestrian count across their own sub-zones (see ZONE_PIE_COLORS); ROAD
// pools every ROAD zone's vehicles by type instead — roads carry little foot traffic, so the ROAD
// pie is vehicle-only (no pedestrian slice/count at all).
static const int PANEL_PIE_RADIUS = 65; // px, 130px diameter — shared by all 3 category pies

// Colors for the CROSSWALK/SIDEWALK per-sub-zone pies, assigned by sorted position within the
// category (e.g. CROSSWALK_NE/NW/S or SIDEWALK_E/N/W) — flat-UI blue/green/orange, distinct from
// the vehicle-type palette used by the ROAD pie.
static const std::vector<cv::Scalar> ZONE_PIE_COLORS = {
    cv::Scalar(219, 152, 52), cv::Scalar(113, 204, 46), cv::Scalar(34, 126, 230),
};

// Line heights for the per-sub-zone (3-line) and ROAD vehicle-type (8-line) legends.
static const int ZONE_LEGEND_LINE_H = 26;
static const int ROAD_LEGEND_LINE_H = 24;

// How far below pieCenter every pie's caption sits (see drawPieCaption) — the same value for all
// 3 pies so their captions line up on one row, using whichever content reaches lowest: the pie's
// own radius, or a legend taller than the pie (ROAD's 8 lines reach further down than
// CROSSWALK/SIDEWALK's 3-line legend or the pie itself).
static const int PANEL_CAPTION_REACH = std::max({PANEL_PIE_RADIUS, (3 - 1) * ZONE_LEGEND_LINE_H / 2,
                                                  (int)(VEHICLE_TYPE_ORDER.size() - 1) * ROAD_LEGEND_LINE_H / 2});

// Full compass names for a zone's directional suffix (e.g. CROSSWALK_NE -> "Northeast"), used in
// the per-sub-zone pie legend instead of the raw abbreviation.
static const std::unordered_map<std::string, std::string> DIRECTION_NAMES = {
    {"N", "North"}, {"S", "South"}, {"E", "East"}, {"W", "West"},
    {"NE", "Northeast"}, {"NW", "Northwest"}, {"SE", "Southeast"}, {"SW", "Southwest"},
};

// Groups ROI names by their category prefix, sorted within each column.
static std::vector<std::vector<std::string>> groupedZones(const ROIMap& roi) {
    std::vector<std::vector<std::string>> grouped;
    for (auto& cat : PANEL_CATEGORIES) {
        std::vector<std::string> names;
        for (auto& [name, poly] : roi)
            if (name.rfind(cat, 0) == 0) names.push_back(name);
        std::sort(names.begin(), names.end());
        grouped.push_back(names);
    }
    return grouped;
}

// Computes the height (px) of the standalone stats strip: fixed, since every column always
// renders exactly one pooled row now (no more per-zone rows of variable count).
static int panelHeight() {
    return PANEL_GRID_Y + PANEL_ROW_H + PANEL_BOTTOM_PAD;
}

// Measures a putText string's pixel width at a given font scale.
static int textW(const std::string& text, double scale) {
    int baseline = 0;
    return cv::getTextSize(text, FONT, scale, 1, &baseline).width;
}

// vehicleTypes mirrors Python's instant[name]["vehicle_types"] dict: per-zone instant count of
// each vehicle type, used to build the ROAD pie (see drawRoadSummaryRow). Always holds one entry
// per VEHICLE_TYPE_ORDER, initialized to 0 by makeZeroCounts() below.
struct Counts { int pedestrian = 0, vehicle = 0; std::unordered_map<std::string, int> vehicleTypes; };
struct UniqueSets { std::set<int> pedestrian, vehicle; };

static Counts makeZeroCounts() {
    Counts c;
    for (auto& vtype : VEHICLE_TYPE_ORDER) c.vehicleTypes[vtype] = 0;
    return c;
}

// Draws a small pie chart at `center`: each category's slice angle is its share of the total
// count. A zone with no vehicles at all gets a plain gray "no data" circle instead of an
// empty/misleading pie. A thin dark outline separates adjacent same-ish-hue slices.
static void drawPieChart(cv::Mat& panel, cv::Point center, int radius,
                          const std::unordered_map<std::string, int>& counts,
                          const std::vector<std::string>& order,
                          const std::unordered_map<std::string, cv::Scalar>& colors) {
    int total = 0;
    for (auto& [k, v] : counts) total += v;
    if (total <= 0) {
        cv::circle(panel, center, radius, PIE_EMPTY_COLOR, -1);
        cv::circle(panel, center, radius, PIE_OUTLINE_COLOR, 1, cv::LINE_AA);
        return;
    }
    double startAngle = -90.0; // 12 o'clock, sweeping clockwise
    for (auto& cat : order) {
        auto it = counts.find(cat);
        int cnt = it != counts.end() ? it->second : 0;
        if (cnt <= 0) continue;
        double endAngle = startAngle + 360.0 * cnt / total;
        cv::ellipse(panel, center, cv::Size(radius, radius), 0, startAngle, endAngle, colors.at(cat), -1);
        startAngle = endAngle;
    }
    cv::circle(panel, center, radius, PIE_OUTLINE_COLOR, 1, cv::LINE_AA);
}

// Draws a caption centered under a pie, naming what it counts (e.g. "counts by vehicle type").
// Shared by the CROSSWALK/SIDEWALK per-sub-zone pies and the ROAD vehicle-type pie, always at the
// same PANEL_CAPTION_REACH below pieCenter so all 3 captions line up on one row instead of each
// sitting right under its own (differently sized) pie/legend.
static void drawPieCaption(cv::Mat& panel, cv::Point pieCenter, const std::string& text) {
    int captionX = pieCenter.x - textW(text, PANEL_CAPTION_SCALE) / 2;
    int captionY = pieCenter.y + PANEL_CAPTION_REACH + 20;
    cv::putText(panel, text, cv::Point(captionX, captionY), FONT, PANEL_CAPTION_SCALE,
                PANEL_TEXT_COLOR, 1, cv::LINE_AA);
}

// Draws the CROSSWALK/SIDEWALK column's single combined row: one pie showing how the category's
// current (this-frame) pedestrian count splits across its own sub-zones (e.g. CROSSWALK_NE vs
// _NW vs _S), with a color-keyed legend beside it (sub-zone's full compass name + its current
// count) and a caption below it naming what the pie counts. Unlike the ROAD pie, every slice here
// is the same "thing" (pedestrians) — the split is by location, not by type — so the slice colors
// are zone-specific (ZONE_PIE_COLORS), not the vehicle-type legend at the top of the panel. Uses
// `instant` (this frame's live count), not `unique` (cumulative-ever-seen), so the pie and its
// legend numbers update live as pedestrians enter/leave the zone.
static void drawZonePieRow(cv::Mat& panel, int colX, int colWidth, int rowY, const std::string& cat,
                           const std::vector<std::string>& zoneNames,
                           std::unordered_map<std::string, Counts>& instant) {
    std::unordered_map<std::string, int> counts;
    std::unordered_map<std::string, cv::Scalar> colors;
    for (size_t i = 0; i < zoneNames.size(); i++) {
        counts[zoneNames[i]] = instant[zoneNames[i]].pedestrian;
        colors[zoneNames[i]] = ZONE_PIE_COLORS[i % ZONE_PIE_COLORS.size()];
    }

    cv::Point pieCenter(colX + colWidth / 2, rowY - 17);
    drawPieChart(panel, pieCenter, PANEL_PIE_RADIUS, counts, zoneNames, colors);

    int legendX = pieCenter.x + PANEL_PIE_RADIUS + PIE_GAP_AFTER;
    int startY = pieCenter.y - (int)(zoneNames.size() - 1) * ZONE_LEGEND_LINE_H / 2;
    for (size_t i = 0; i < zoneNames.size(); i++) {
        auto& name = zoneNames[i];
        int y = startY + (int)i * ZONE_LEGEND_LINE_H;
        cv::rectangle(panel, cv::Point(legendX, y - 13), cv::Point(legendX + 16, y + 3), colors[name], -1);
        std::string suffix = name.substr(name.find('_') + 1);
        auto dirIt = DIRECTION_NAMES.find(suffix);
        std::string direction = dirIt != DIRECTION_NAMES.end() ? dirIt->second : name;
        std::string line = direction + " " + std::to_string(counts[name]);
        cv::putText(panel, line, cv::Point(legendX + 24, y), FONT, PANEL_ZONE_LEGEND_SCALE,
                    PANEL_TEXT_COLOR, 1, cv::LINE_AA);
    }

    std::string catLower = cat;
    std::transform(catLower.begin(), catLower.end(), catLower.begin(), ::tolower);
    drawPieCaption(panel, pieCenter, catLower + " count of pedestrians");
}

// Draws the ROAD column's single combined row: all ROAD zones pooled together into one bigger pie
// broken down by vehicle type only (no pedestrian slice/count) instead of one row per zone — roads
// carry little foot traffic, so pedestrian activity there isn't tracked in this column at all.
// vehicleTypes is already summed across every ROAD zone. A color-keyed legend to the right of the
// pie names every vehicle type with its count (mirrors drawZonePieRow's per-sub-zone legend), and
// a caption below names what the pie counts (see drawPieCaption).
static void drawRoadSummaryRow(cv::Mat& panel, int colX, int colWidth, int rowY,
                                const std::unordered_map<std::string, int>& vehicleTypes) {
    // Centered in the column's full width, since there's only ever one pie here.
    cv::Point pieCenter(colX + colWidth / 2, rowY - 17);
    drawPieChart(panel, pieCenter, PANEL_PIE_RADIUS, vehicleTypes, VEHICLE_TYPE_ORDER, VEHICLE_TYPE_COLORS);

    int legendX = pieCenter.x + PANEL_PIE_RADIUS + PIE_GAP_AFTER;
    int startY = pieCenter.y - (int)(VEHICLE_TYPE_ORDER.size() - 1) * ROAD_LEGEND_LINE_H / 2;
    for (size_t i = 0; i < VEHICLE_TYPE_ORDER.size(); i++) {
        auto& vtype = VEHICLE_TYPE_ORDER[i];
        int y = startY + (int)i * ROAD_LEGEND_LINE_H;
        cv::rectangle(panel, cv::Point(legendX, y - 13), cv::Point(legendX + 16, y + 3),
                      VEHICLE_TYPE_COLORS.at(vtype), -1);
        std::string line = vtype + " " + std::to_string(vehicleTypes.at(vtype));
        cv::putText(panel, line, cv::Point(legendX + 24, y), FONT, PANEL_ZONE_LEGEND_SCALE,
                    PANEL_TEXT_COLOR, 1, cv::LINE_AA);
    }

    drawPieCaption(panel, pieCenter, "counts by vehicle type");
}

// Renders the live statistics strip as its own image, stacked above the video frame rather than
// overlaid on top of it (no intersection, source video resolution/aspect ratio stays untouched).
// Zones are grouped into 3 columns (crosswalk / sidewalk / road) spread across the full frame
// width; each column renders a single pooled pie row — see drawZonePieRow (CROSSWALK/SIDEWALK,
// split by sub-zone, this frame's live counts) and drawRoadSummaryRow (ROAD, vehicle types only,
// no pedestrian data).
static cv::Mat drawPanel(int width, const ROIMap& roi,
                         std::unordered_map<std::string, Counts>& instant) {
    int y0 = PANEL_Y0;
    auto grouped = groupedZones(roi);
    int height = panelHeight();
    cv::Mat panel(height, width, CV_8UC3, cv::Scalar(15, 15, 15));
    int colWidth = (width - 2 * PANEL_MARGIN) / (int)PANEL_CATEGORIES.size();

    // Dashboard title, centered across the full panel width above everything else.
    int titleX = (width - textW(PANEL_TITLE, PANEL_TITLE_SCALE)) / 2;
    cv::putText(panel, PANEL_TITLE, cv::Point(titleX, PANEL_TITLE_Y), FONT, PANEL_TITLE_SCALE,
                PANEL_TEXT_COLOR, 2, cv::LINE_AA);

    // Single-line legend: every vehicle type — the colors used by the ROAD pie (CROSSWALK/SIDEWALK
    // pies use their own per-sub-zone legend, see drawZonePieRow) — horizontally centered so the
    // left/right margins match. Each item's footprint is swatch(22) + gap-to-text(8) + text +
    // trailing-gap-to-next-item(30); the last item's trailing gap isn't part of the visible
    // content, hence the "- 30" below.
    int legendContentW = -30;
    for (auto& vtype : VEHICLE_TYPE_ORDER) legendContentW += 60 + textW(vtype, PANEL_LEGEND_SCALE);
    int xCursor = std::max(PANEL_MARGIN, (width - legendContentW) / 2);
    for (auto& vtype : VEHICLE_TYPE_ORDER) {
        cv::Scalar color = VEHICLE_TYPE_COLORS.at(vtype);
        cv::rectangle(panel, cv::Point(xCursor, y0 + 4), cv::Point(xCursor + 22, y0 + 28), color, -1);
        int labelX = xCursor + 30;
        cv::putText(panel, vtype, cv::Point(labelX, y0 + 26), FONT, PANEL_LEGEND_SCALE,
                    PANEL_TEXT_COLOR, 1, cv::LINE_AA);
        xCursor = labelX + textW(vtype, PANEL_LEGEND_SCALE) + 30;
    }

    for (size_t col = 0; col < PANEL_CATEGORIES.size(); col++) {
        int colX = PANEL_MARGIN + (int)col * colWidth;
        auto& cat = PANEL_CATEGORIES[col];
        int headerX = colX + (colWidth - textW(cat, PANEL_HEADER_SCALE)) / 2;
        cv::putText(panel, cat, cv::Point(headerX, PANEL_HEADER_Y), FONT, PANEL_HEADER_SCALE,
                    PANEL_TEXT_COLOR, 2, cv::LINE_AA);
    }

    int gridY0 = PANEL_GRID_Y;
    for (size_t col = 0; col < PANEL_CATEGORIES.size(); col++) {
        int colX = PANEL_MARGIN + (int)col * colWidth;
        auto& cat = PANEL_CATEGORIES[col];
        if (cat == "ROAD") {
            auto& roadNames = grouped[col];
            std::unordered_map<std::string, int> aggVehicleTypes;
            for (auto& t : VEHICLE_TYPE_ORDER) {
                int sum = 0;
                for (auto& n : roadNames) sum += instant[n].vehicleTypes.at(t);
                aggVehicleTypes[t] = sum;
            }
            drawRoadSummaryRow(panel, colX, colWidth, gridY0, aggVehicleTypes);
        } else {
            drawZonePieRow(panel, colX, colWidth, gridY0, cat, grouped[col], instant);
        }
    }

    return panel;
}

// Draws the ROIs on the first frame, saves a PNG and prints click coordinates.
static void roiPreview(const fs::path& videoPath, const ROIMap& roi) {
    cv::VideoCapture cap(videoPath.string());
    cv::Mat frame;
    bool ok = cap.isOpened() && cap.read(frame);
    cap.release();
    if (!ok) {
        std::cerr << "[ERROR] Could not read the first frame: " << videoPath << "\n";
        std::exit(1);
    }

    frame = drawRois(frame, roi, 0.22);
    for (auto& [name, poly] : roi)
        for (auto& p : poly)
            cv::circle(frame, p, 4, cv::Scalar(255, 255, 255), -1);

    fs::path output = videoPath.parent_path() / (videoPath.stem().string() + "_roi_preview.png");
    cv::imwrite(output.string(), frame);
    std::cout << "[OK] ROI preview saved: " << output << "\n";

    try {
        double scale = 0.66;
        cv::Mat display;
        cv::resize(frame, display, cv::Size(), scale, scale);
        std::string window = "ROI Preview | left click: coordinate | q: quit";

        struct ClickCtx { double scale; };
        static ClickCtx ctx{scale};
        cv::namedWindow(window);
        cv::setMouseCallback(window, [](int event, int x, int y, int, void* userdata) {
            if (event == cv::EVENT_LBUTTONDOWN) {
                double s = ((ClickCtx*)userdata)->scale;
                std::cout << "    Clicked point: [" << (int)(x / s) << ", " << (int)(y / s) << "]\n";
            }
        }, &ctx);
        cv::imshow(window, display);
        while (true) {
            int key = cv::waitKey(50) & 0xFF;
            if (key == 'q' || key == 27) break;
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        }
        cv::destroyAllWindows();
    } catch (const cv::Exception&) {
        std::cout << "[INFO] Window could not be opened (headless environment?); check the saved PNG.\n";
    }
}

// Minimal JSON emission helpers (structure/keys mirror the Python summary dict exactly;
// numeric formatting doesn't need to match Python's repr byte-for-byte, only the value).
static std::string jsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}
static std::string jNum(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

// ----------------------------------- main -----------------------------------
int main() {
    fs::path videoPath = VIDEO_PATH;
    if (!fs::exists(videoPath)) {
        std::cerr << "[ERROR] Video not found: " << videoPath << "\n";
        return 1;
    }

    cv::VideoCapture videoCapture(videoPath.string());
    if (!videoCapture.isOpened()) {
        std::cerr << "[ERROR] Could not open video: " << videoPath << "\n";
        return 1;
    }

    int W = (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = videoCapture.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 60.0;
    int totalFrames = (int)videoCapture.get(cv::CAP_PROP_FRAME_COUNT);

    ROIMap roi = scaleRois(makeRoi(), W, H);
    std::vector<std::string> crosswalkNames;
    for (auto& [name, poly] : roi)
        if (name.rfind("CROSSWALK", 0) == 0) crosswalkNames.push_back(name);
    int PANEL_H = panelHeight();
    int outH = H + PANEL_H; // output frame height: stats strip on top + video below, no overlap

    if (CHECK_ROI) {
        videoCapture.release();
        roiPreview(videoPath, roi);
        return 0;
    }

    fs::path modelPath = MODEL_PATH;
    if (modelPath.filename().string().rfind("visdrone", 0) == 0 && !fs::exists(modelPath)) {
        std::cerr << "[ERROR] Model weights not found: " << modelPath << "\n"
                  << "        Export the ONNX weights from the VisDrone model on the "
                     "Ultralytics Platform and\n"
                  << "        save them next to this executable as 'visdrone.onnx', "
                     "or set MODEL_PATH to the full path.\n";
        return 1;
    }

    std::cout << "[INFO] Model: " << MODEL_PATH << " | Device: " << DEVICE << " | "
              << W << "x" << H << " @ " << std::fixed << std::setprecision(2) << fps
              << " FPS | " << totalFrames << " frames | " << roi.size() << " ROI\n";

    Detector detector(MODEL_PATH, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, trackClasses(), DEVICE);

    BotSortParams trackerParams;
    trackerParams.frameRate = fps;
    BotSort tracker(trackerParams);

    fs::path outputVideoPath = videoPath.parent_path() / (videoPath.stem().string() + "-Analysis.mp4");
    fs::path outputCsvPath = videoPath.parent_path() / (videoPath.stem().string() + "-timeseries.csv");
    fs::path outputJsonPath = videoPath.parent_path() / (videoPath.stem().string() + "-summary.json");

    cv::VideoWriter outputVideo(outputVideoPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                std::max(fps / STRIDE, 1.0), cv::Size(W, outH));

    bool show = SHOW_PREVIEW;
    std::string window = "Vehicle and Pedestrian Traffic Analysis System";
    if (show) {
        try {
            // Scale the preview window to fit the screen; the saved video file keeps full resolution
            double previewScale = std::min({1.0, (double)MAX_PREVIEW_WIDTH / W, (double)MAX_PREVIEW_HEIGHT / outH});
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, std::max(1, (int)(W * previewScale)), std::max(1, (int)(outH * previewScale)));
        } catch (const cv::Exception&) {
            std::cout << "[INFO] Preview window could not be opened (headless?); "
                         "processing continues without preview, the video is still saved.\n";
            show = false;
        }
    }

    std::ofstream csvFile(outputCsvPath.string(), std::ios::binary);
    {
        std::string header = "frame,time_sec";
        for (auto& [name, poly] : roi) header += "," + name + "_pedestrian," + name + "_vehicle";
        header += ",frame_pedestrian_instant,frame_vehicle_instant";
        csvFile << header << "\r\n";
    }

    // State variables
    std::unordered_map<std::string, Counts> instant, peak;
    std::unordered_map<std::string, UniqueSets> unique;
    std::map<std::tuple<std::string, int, std::string>, int> dwellFrames; // (zone, tid, category) -> processed frames
    std::set<int> framePedestrianIds, frameVehicleIds;
    std::map<std::string, std::set<int>> personTypes;   // person class (pedestrian/people) -> {tid}
    std::map<std::string, std::set<int>> vehicleTypeIds; // type name -> {tid}
    std::unordered_map<int, std::deque<cv::Point2f>> motionHistory; // tid -> recent centers

    for (auto& [name, poly] : roi) {
        instant[name] = makeZeroCounts();
        unique[name] = {};
        peak[name] = {};
    }

    int frameIdx = -1, processed = 0, lastLogSec = -1;
    auto t0 = std::chrono::steady_clock::now();
    double processingFps = 0.0;

    cv::Mat frame;
    while (true) {
        if (!videoCapture.read(frame)) break;
        frameIdx++;
        if (STRIDE > 1 && frameIdx % STRIDE) continue;
        if (MAX_FRAMES && processed >= MAX_FRAMES) break;
        processed++;
        double timeSec = frameIdx / fps;

        std::vector<Detection> detections = detector.detect(frame);
        auto tracks = tracker.update(detections, frame);

        for (auto& [name, poly] : roi) instant[name] = makeZeroCounts();
        int framePedestrianInstant = 0, frameVehicleInstant = 0;

        cv::Mat annotatedFrame = frame.clone();
        struct DrawBox { cv::Point p1, p2; cv::Scalar color; };
        std::vector<DrawBox> boxesToDraw;
        std::unordered_map<std::string, int> movingPedestrianCounts; // zone -> moving pedestrians in this frame
        std::set<std::string> crosswalkVehicleHits; // crosswalk zone names whose polygon border/area is
                                                      // touched by any vehicle's detection box this frame

        for (auto& track : tracks) {
            int tid = track->trackId;
            int cid = track->cls;
            cv::Rect2f b = track->tlbr(); // (x1, y1, x2, y2) packed as (x, y, width, height)
            float x1 = b.x, y1 = b.y, x2 = b.width, y2 = b.height;

            cv::Point2f base((x1 + x2) / 2.0f, y2); // ground contact point (bottom-center)
            bool isPerson = PERSON_CLASSES.count(cid) > 0;
            std::string category = isPerson ? "pedestrian" : "vehicle";

            if (isPerson) {
                framePedestrianInstant++;
                framePedestrianIds.insert(tid);
                personTypes[PERSON_CLASSES.at(cid)].insert(tid);
            } else {
                std::string vtype = VEHICLE_CLASSES.at(cid);
                frameVehicleInstant++;
                frameVehicleIds.insert(tid);
                vehicleTypeIds[vtype].insert(tid);
                // Any border of a motorized vehicle's box touching a crosswalk polygon marks that
                // crosswalk red below, regardless of where the ground-contact point lands. bicycle
                // is the only excluded type.
                if (MOTORIZED_VEHICLE_TYPES.count(vtype)) {
                    for (auto& cwName : crosswalkNames) {
                        if (crosswalkVehicleHits.count(cwName)) continue;
                        for (auto& [n, poly] : roi) {
                            if (n != cwName) continue;
                            if (boxIntersectsPolygon(x1, y1, x2, y2, poly)) crosswalkVehicleHits.insert(cwName);
                            break;
                        }
                    }
                }
            }

            auto zone = findZone(base, roi);
            if (zone) {
                if (category == "pedestrian") instant[*zone].pedestrian++; else instant[*zone].vehicle++;
                if (category == "pedestrian") unique[*zone].pedestrian.insert(tid); else unique[*zone].vehicle.insert(tid);
                dwellFrames[{*zone, tid, category}]++;
                if (!isPerson) instant[*zone].vehicleTypes[VEHICLE_CLASSES.at(cid)]++;
            }

            // Exact box center (not the ground-contact point used for zones): the correct
            // reference for motion, since the ground point jitters with perspective/pose.
            cv::Point2f boxCenter((x1 + x2) / 2.0f, (y1 + y2) / 2.0f);
            bool moving = isMoving(tid, boxCenter, motionHistory, MOTION_THRESH);
            cv::Scalar boxColor = moving ? MOVING_COLOR : STATIONARY_COLOR;

            if (zone && isPerson && moving) movingPedestrianCounts[*zone]++;

            boxesToDraw.push_back({cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), boxColor});
        }

        // Crosswalks get a color mask: blue if pedestrians outnumber motorized vehicles in that
        // zone by PEDESTRIAN_OVERRIDE_MARGIN+ (pedestrians clearly own the crossing, so red is
        // suppressed even with a vehicle inside), else red if any vehicle's detection box touches
        // the polygon's border or area, else blue if MIN_MOVING_PEDESTRIANS+ moving pedestrians
        // are inside.
        std::unordered_map<std::string, cv::Scalar> crosswalkZoneColors;
        for (auto& name : crosswalkNames) {
            int motorizedCount = 0;
            for (auto& vtype : MOTORIZED_VEHICLE_TYPES) motorizedCount += instant[name].vehicleTypes.at(vtype);
            int pedestrianLead = instant[name].pedestrian - motorizedCount;
            if (pedestrianLead >= PEDESTRIAN_OVERRIDE_MARGIN) {
                crosswalkZoneColors[name] = CROSSWALK_PEDESTRIAN_COLOR;
            } else if (crosswalkVehicleHits.count(name)) {
                crosswalkZoneColors[name] = CROSSWALK_VEHICLE_COLOR;
            } else if (movingPedestrianCounts[name] >= MIN_MOVING_PEDESTRIANS) {
                crosswalkZoneColors[name] = CROSSWALK_PEDESTRIAN_COLOR;
            }
        }
        annotatedFrame = drawCrosswalkMasks(annotatedFrame, roi, crosswalkZoneColors);

        if (SHOW_BOXES) {
            for (auto& box : boxesToDraw) cv::rectangle(annotatedFrame, box.p1, box.p2, box.color, 2);
        }

        for (auto& [name, poly] : roi) {
            peak[name].pedestrian = std::max(peak[name].pedestrian, instant[name].pedestrian);
            peak[name].vehicle = std::max(peak[name].vehicle, instant[name].vehicle);
        }

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        processingFps = processed / std::max(elapsed, 1e-6);
        cv::Mat panel = drawPanel(W, roi, instant);
        cv::vconcat(panel, annotatedFrame, annotatedFrame); // strip above the frame, no overlap

        // Write the drawn frame to the video file to be saved (before showing, so it's kept even if stopped with 'q').
        outputVideo.write(annotatedFrame);

        if (show) {
            cv::imshow(window, annotatedFrame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                std::cout << "[INFO] Stopped by user; outputs so far are being saved.\n";
                break;
            }
            if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                std::cout << "[INFO] Preview closed; processing continues in the background.\n";
                show = false;
            }
        }

        // One time-series row per second.
        int sec = (int)timeSec;
        if (sec != lastLogSec) {
            lastLogSec = sec;
            std::ostringstream row;
            row << frameIdx << "," << std::fixed << std::setprecision(2) << timeSec;
            for (auto& [name, poly] : roi) row << "," << instant[name].pedestrian << "," << instant[name].vehicle;
            row << "," << framePedestrianInstant << "," << frameVehicleInstant;
            csvFile << row.str() << "\r\n";
        }

        if (processed % 300 == 0) {
            double percent = 100.0 * frameIdx / std::max(totalFrames, 1);
            std::printf("[%5.1f%%] frame %d/%d | video t=%6.1f s | proc %.1f FPS\n",
                        percent, frameIdx, totalFrames, timeSec, processingFps);
        }
    }

    videoCapture.release();
    outputVideo.release();
    csvFile.close();
    try { cv::destroyAllWindows(); } catch (const cv::Exception&) {}

    // Computes the average dwell time (seconds) for a zone/category.
    auto avgDwell = [&](const std::string& name, const std::string& category) -> double {
        std::vector<double> durations;
        for (auto& [key, n] : dwellFrames) {
            auto& [z, t, k] = key;
            if (z == name && k == category) durations.push_back(n * (double)STRIDE / fps);
        }
        if (durations.empty()) return 0.0;
        double sum = 0;
        for (double d : durations) sum += d;
        return std::round(sum / durations.size() * 100.0) / 100.0;
    };

    std::ostringstream json;
    json << "{\n";
    json << "  \"video\": \"" << jsonEscape(videoPath.filename().string()) << "\",\n";
    json << "  \"resolution\": \"" << W << "x" << H << "\",\n";
    json << "  \"fps\": " << jNum(fps) << ",\n";
    json << "  \"processed_frames\": " << processed << ",\n";
    json << "  \"analysis_duration_sec\": " << jNum(frameIdx >= 0 ? frameIdx / fps : 0.0) << ",\n";
    json << "  \"model\": \"" << jsonEscape(MODEL_PATH) << "\",\n";
    json << "  \"conf\": " << CONF_THRESHOLD << ",\n";
    json << "  \"iou\": " << IOU_THRESHOLD << ",\n";
    json << "  \"motion_thresh\": " << MOTION_THRESH << ",\n";
    json << "  \"motion_window\": " << MOTION_WINDOW << ",\n";
    json << "  \"zones\": {\n";
    bool firstZone = true;
    for (auto& [name, poly] : roi) {
        if (!firstZone) json << ",\n";
        firstZone = false;
        json << "    \"" << name << "\": {\n"
             << "      \"unique_pedestrian\": " << unique[name].pedestrian.size() << ",\n"
             << "      \"unique_vehicle\": " << unique[name].vehicle.size() << ",\n"
             << "      \"peak_instant_pedestrian\": " << peak[name].pedestrian << ",\n"
             << "      \"peak_instant_vehicle\": " << peak[name].vehicle << ",\n"
             << "      \"avg_dwell_sec_pedestrian\": " << jNum(avgDwell(name, "pedestrian")) << ",\n"
             << "      \"avg_dwell_sec_vehicle\": " << jNum(avgDwell(name, "vehicle")) << "\n"
             << "    }";
    }
    json << "\n  },\n";
    json << "  \"frame_overall\": {\n"
         << "    \"unique_pedestrian\": " << framePedestrianIds.size() << ",\n"
         << "    \"unique_vehicle\": " << frameVehicleIds.size() << ",\n"
         << "    \"person_type_distribution\": {";
    {
        bool first = true;
        for (auto& [t, ids] : personTypes) {
            if (!first) json << ", ";
            first = false;
            json << "\"" << jsonEscape(t) << "\": " << ids.size();
        }
    }
    json << "},\n"
         << "    \"vehicle_type_distribution\": {";
    {
        bool first = true;
        for (auto& [t, ids] : vehicleTypeIds) {
            if (!first) json << ", ";
            first = false;
            json << "\"" << jsonEscape(t) << "\": " << ids.size();
        }
    }
    json << "}\n  }\n}\n";

    std::ofstream jsonFile(outputJsonPath.string(), std::ios::binary);
    jsonFile << json.str();
    jsonFile.close();

    std::cout << "\n================================ SUMMARY ================================\n";
    for (auto& [name, poly] : roi) {
        std::printf("%-14s pedestrian: %5d unique | peak %3d | avg %5.1f s  ||  vehicle: %4d | peak %2d | avg %4.1f s\n",
                    name.c_str(), (int)unique[name].pedestrian.size(), peak[name].pedestrian,
                    avgDwell(name, "pedestrian"), (int)unique[name].vehicle.size(), peak[name].vehicle,
                    avgDwell(name, "vehicle"));
    }
    {
        std::ostringstream personDist, vehicleDist;
        personDist << "{";
        bool first = true;
        for (auto& [t, ids] : personTypes) { if (!first) personDist << ", "; first = false; personDist << "'" << t << "': " << ids.size(); }
        personDist << "}";
        vehicleDist << "{";
        first = true;
        for (auto& [t, ids] : vehicleTypeIds) { if (!first) vehicleDist << ", "; first = false; vehicleDist << "'" << t << "': " << ids.size(); }
        vehicleDist << "}";
        std::printf("%-14s pedestrian: %zu unique %s | vehicle: %zu %s\n", "FRAME",
                    framePedestrianIds.size(), personDist.str().c_str(),
                    frameVehicleIds.size(), vehicleDist.str().c_str());
    }
    std::cout << "========================================================================\n";
    std::cout << "[OK] Video : " << outputVideoPath << "\n";
    std::cout << "[OK] CSV   : " << outputCsvPath << "\n";
    std::cout << "[OK] JSON  : " << outputJsonPath << "\n";

    return 0;
}
