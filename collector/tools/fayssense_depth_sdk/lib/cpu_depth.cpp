/**
 * CPU-only stereo fisheye depth pipeline (C++ port of benchmark/).
 *
 * Variants:
 *   V1  cv::fisheye::stereoRectify pinhole rectification + SGBM
 *   V2v vertical-polar ERP rectification + SGBM
 *   V2b baseline-polar ERP rectification + SGBM
 *   V3  multi-pinhole bands + relaxed SGBM + ERP fusion
 *   V4  pinhole rectification + StereoBM (fast)
 *   V5  engine-rectified input + own SGBM
 */

#include "cpu_depth.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace cpu_depth {

namespace {

const double kPi = 3.14159265358979323846;

inline double deg2rad(double d) { return d * kPi / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / kPi; }

// ---------------------------------------------------------------------------
// kalibr yaml parsing (line based, no yaml-cpp dependency)
// ---------------------------------------------------------------------------

std::string trimLine(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r");
    size_t b = s.find_last_not_of(" \t\r");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

/// Extract the first `[a, b, c, ...]` list of doubles from a yaml line.
std::vector<double> parseList(const std::string& line)
{
    std::vector<double> v;
    size_t i = line.find('[');
    if (i == std::string::npos) return v;
    size_t j = line.find(']', i);
    if (j == std::string::npos) j = line.size();
    size_t p = i + 1;
    while (p < j) {
        while (p < j && (line[p] == ' ' || line[p] == ',' || line[p] == '\t')) ++p;
        if (p >= j) break;
        char* end = nullptr;
        double val = std::strtod(line.c_str() + p, &end);
        if (end == line.c_str() + p) break;   // not a number
        v.push_back(val);
        p = end - line.c_str();
    }
    return v;
}

// ---------------------------------------------------------------------------
// KB4 (equidistant) projection
// ---------------------------------------------------------------------------

/// KB4 forward projection: camera point -> pixel.
void kb4Project(const cv::Vec3d& p, const cv::Matx33d& K, const cv::Vec4d& D,
                double& u, double& v)
{
    double rho = std::sqrt(p[0] * p[0] + p[1] * p[1]);
    double th = std::atan2(rho, p[2]);
    double t2 = th * th, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
    double td = th * (1.0 + D[0] * t2 + D[1] * t4 + D[2] * t6 + D[3] * t8);
    double safe = rho > 1e-12 ? rho : 1.0;
    u = K(0, 0) * td * p[0] / safe + K(0, 2);
    v = K(1, 1) * td * p[1] / safe + K(1, 2);
}

/// KB4 inverse: pixel -> unit ray direction (Newton on the odd polynomial).
cv::Vec3d kb4BackProject(double u, double v, const cv::Matx33d& K, const cv::Vec4d& D)
{
    double xn = (u - K(0, 2)) / K(0, 0);
    double yn = (v - K(1, 2)) / K(1, 1);
    double td = std::sqrt(xn * xn + yn * yn);
    double th = td;
    for (int it = 0; it < 10; ++it) {
        double t2 = th * th, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
        double f = th * (1.0 + D[0] * t2 + D[1] * t4 + D[2] * t6 + D[3] * t8) - td;
        double fp = 1.0 + 3 * D[0] * t2 + 5 * D[1] * t4 + 7 * D[2] * t6 + 9 * D[3] * t8;
        th -= f / std::max(std::abs(fp), 1e-9);
    }
    th = std::min(std::max(th, 0.0), kPi);
    double sinT = std::sin(th), cosT = std::cos(th);
    double rho = td > 1e-12 ? sinT / td : 0.0;
    return cv::Vec3d(rho * xn, rho * yn, td > 1e-12 ? cosT : 1.0);
}

// ---------------------------------------------------------------------------
// Map building
// ---------------------------------------------------------------------------

/// Build a u/v pixel grid (CV_32FC1 each).
void buildGrid(int W, int H, cv::Mat& u, cv::Mat& v)
{
    u.create(H, W, CV_32FC1);
    v.create(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        float* up = u.ptr<float>(y);
        float* vp = v.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            up[x] = static_cast<float>(x);
            vp[x] = static_cast<float>(y);
        }
    }
}

/// Back-project rectified rays into both fisheye cams -> remap map pairs + masks.
/// Rays outside the source image or beyond the lens FOV become -1 in the maps.
void raysToRemapPair(const cv::Mat& rays /*Nx3 CV_32F, rectified frame*/,
                     const StereoCalib& calib, int outW, int outH,
                     cv::Mat& mapx0, cv::Mat& mapy0, cv::Mat& mapx1, cv::Mat& mapy1,
                     cv::Mat& valid0, cv::Mat& valid1)
{
    const int N = rays.rows;
    mapx0.create(outH, outW, CV_32FC1);
    mapy0.create(outH, outW, CV_32FC1);
    mapx1.create(outH, outW, CV_32FC1);
    mapy1.create(outH, outW, CV_32FC1);
    valid0.create(outH, outW, CV_8UC1);
    valid1.create(outH, outW, CV_8UC1);

    for (int i = 0; i < N; ++i) {
        const float* r = rays.ptr<float>(i);
        cv::Vec3d v(r[0], r[1], r[2]);
        cv::Vec3d d0 = calib.R_rect.t() * v;   // rectified -> cam0
        cv::Vec3d d1 = calib.R1_rect.t() * v;  // rectified -> cam1
        double u0, v0, u1, v1;
        kb4Project(d0, calib.K[0], calib.D[0], u0, v0);
        kb4Project(d1, calib.K[1], calib.D[1], u1, v1);
        double th0 = std::acos(std::min(std::max(d0[2], -1.0), 1.0));
        double th1 = std::acos(std::min(std::max(d1[2], -1.0), 1.0));
        bool ok0 = u0 >= 0 && u0 < calib.frameW && v0 >= 0 && v0 < calib.frameH
                && th0 <= calib.lensMaxTheta;
        bool ok1 = u1 >= 0 && u1 < calib.frameW && v1 >= 0 && v1 < calib.frameH
                && th1 <= calib.lensMaxTheta;
        mapx0.at<float>(i) = ok0 ? static_cast<float>(u0) : -1.0f;
        mapy0.at<float>(i) = ok0 ? static_cast<float>(v0) : -1.0f;
        mapx1.at<float>(i) = ok1 ? static_cast<float>(u1) : -1.0f;
        mapy1.at<float>(i) = ok1 ? static_cast<float>(v1) : -1.0f;
        valid0.at<uchar>(i) = ok0 ? 1 : 0;
        valid1.at<uchar>(i) = ok1 ? 1 : 0;
    }
}

void remapPair(const cv::Mat& left, const cv::Mat& right,
               const cv::Mat& mapx0, const cv::Mat& mapy0,
               const cv::Mat& mapx1, const cv::Mat& mapy1,
               cv::Mat& rL, cv::Mat& rR)
{
    cv::remap(left, rL, mapx0, mapy0, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::remap(right, rR, mapx1, mapy1, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT, cv::Scalar(0));
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

cv::Mat toGray(const cv::Mat& m)
{
    if (m.channels() == 1) return m;
    cv::Mat g;
    cv::cvtColor(m, g, cv::COLOR_BGR2GRAY);
    return g;
}

/// SGBM -> float32 disparity in px. `relaxed` uses the benchmark's band preset
/// (small block, no uniqueness/speckle filter, P1/P2 from the block size).
/// Note: the GUI's WLS checkbox is ignored here — ximgproc may not be linked.
cv::Mat sgbmMatch(const cv::Mat& L, const cv::Mat& R, const CpuOptions& o, bool relaxed)
{
    cv::Mat gL = toGray(L), gR = toGray(R);
    int bs = relaxed ? 5 : o.blockSize;
    if (bs % 2 == 0) ++bs;
    int numDisp = std::max(16, (o.numDisp / 16) * 16);
    int P1 = o.P1, P2 = o.P2;
    if (relaxed) {
        P1 = 8 * 3 * bs * bs;
        P2 = 32 * 3 * bs * bs;
    }
    int uniq = relaxed ? 0 : o.uniquenessRatio;
    int sw = relaxed ? 0 : o.speckleWindowSize;
    int sr = relaxed ? 0 : o.speckleRange;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        o.minDisp, numDisp, bs, P1, P2, o.disp12MaxDiff, o.preFilterCap,
        uniq, sw, sr, o.sgbmMode);
    cv::Mat d16, d;
    sgbm->compute(gL, gR, d16);
    d16.convertTo(d, CV_32FC1, 1.0 / 16.0);
    return d;
}

cv::Mat bmMatch(const cv::Mat& L, const cv::Mat& R, const CpuOptions& o)
{
    cv::Mat gL = toGray(L), gR = toGray(R);
    int numDisp = std::max(16, (o.numDisp / 16) * 16);
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisp, o.bmBlockSize);
    bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    bm->setPreFilterSize(9);
    bm->setPreFilterCap(31);
    bm->setTextureThreshold(20);
    bm->setUniquenessRatio(10);
    cv::Mat d16, d;
    bm->compute(gL, gR, d16);
    d16.convertTo(d, CV_32FC1, 1.0 / 16.0);
    return d;
}

// ---------------------------------------------------------------------------
// Depth conversion
// ---------------------------------------------------------------------------

/// V1/V4/V5: z = fx*B/d on a rectified pinhole pair. 0 = invalid.
void disparityToDepth(const cv::Mat& disp, double fx, double baseline, double maxZ,
                      cv::Mat& depth)
{
    depth.create(disp.size(), CV_32FC1);
    const double scale = fx * baseline;
    for (int y = 0; y < disp.rows; ++y) {
        const float* dp = disp.ptr<float>(y);
        float* zp = depth.ptr<float>(y);
        for (int x = 0; x < disp.cols; ++x) {
            float d = dp[x];
            float z = d > 0.5f ? static_cast<float>(scale / d) : 0.0f;
            zp[x] = z > maxZ ? 0.0f : z;
        }
    }
}

/// V2/V3: per-pixel ray triangulation in the rectified frame.
/// s = ((t x u_r) . (u_l x u_r)) / |u_l x u_r|^2 ; depth along the left ray.
void triangulate(const cv::Mat& raysRect /*Nx3 CV_32F*/,
                 const cv::Mat& disp /*HxW CV_32F px*/,
                 const cv::Vec3d& tRect, double maxZ, double minD,
                 cv::Mat& depth, cv::Mat& pts /*Nx3 CV_32F, zeros for invalid*/)
{
    const int H = disp.rows, W = disp.cols;
    depth = cv::Mat::zeros(H, W, CV_32FC1);
    pts = cv::Mat::zeros(H * W, 3, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        const float* dp = disp.ptr<float>(y);
        float* zp = depth.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            float d = dp[x];
            int xr = static_cast<int>(std::lround(x - d));
            bool ok = d > minD && xr >= 0 && xr < W;
            if (!ok) { zp[x] = 0.0f; continue; }
            const float* ul = raysRect.ptr<float>(y * W + x);
            const float* ur = raysRect.ptr<float>(y * W + xr);
            cv::Vec3d uL(ul[0], ul[1], ul[2]), uR(ur[0], ur[1], ur[2]);
            cv::Vec3d cr = uL.cross(uR);
            double denom = cr.dot(cr);
            cv::Vec3d tcr = tRect.cross(uR);
            double s = tcr.dot(cr) / std::max(denom, 1e-9);
            if (s > 0 && s < maxZ) {
                zp[x] = static_cast<float>(s);
                float* pp = pts.ptr<float>(y * W + x);
                pp[0] = static_cast<float>(s * uL[0]);
                pp[1] = static_cast<float>(s * uL[1]);
                pp[2] = static_cast<float>(s * uL[2]);
            } else {
                zp[x] = 0.0f;  // non-positive, behind-baseline or beyond maxZ
            }
        }
    }
}

// ---------------------------------------------------------------------------
// V3 fusion
// ---------------------------------------------------------------------------

struct BandRes {
    cv::Mat depth;   // band-size CV_32FC1 (m, 0 = invalid)
    cv::Mat pts;     // band N x 3 CV_32FC1, rectified frame
    cv::Mat valid;   // band-size CV_8UC1 (both cameras' rays inside the lens)
    cv::Mat rectL;   // band rectified BGR
    cv::Mat rectR;
};

/// Fuse band results into one ERP depth map (z-buffer, nearest wins), then
/// gap-fill holes up to fillGap px with the nearest valid pixel (depth,
/// band id and source pixel all propagate), then medianBlur(3).
void fuseToErp(const std::vector<BandRes>& bands, int erpW, int erpH,
               double lonDeg, double latDeg, double fillGap,
               cv::Mat& fused, cv::Mat& bandId, cv::Mat& srcPix)
{
    fused = cv::Mat::zeros(erpH, erpW, CV_32FC1);
    bandId = cv::Mat(erpH, erpW, CV_32S, cv::Scalar(-1));
    srcPix = cv::Mat(erpH, erpW, CV_32S, cv::Scalar(-1));
    const double lonMax = deg2rad(lonDeg / 2.0);
    const double latMax = deg2rad(latDeg / 2.0);

    for (size_t b = 0; b < bands.size(); ++b) {
        const cv::Mat& d = bands[b].depth;
        const cv::Mat& p = bands[b].pts;
        const cv::Mat& v = bands[b].valid;
        const int BW = d.cols, BH = d.rows;
        for (int y = 0; y < BH; ++y) {
            const float* dp = d.ptr<float>(y);
            const uchar* vp = v.ptr<uchar>(y);
            for (int x = 0; x < BW; ++x) {
                if (dp[x] <= 0 || !vp[x]) continue;
                const float* pp = p.ptr<float>(y * BW + x);
                if (pp[0] == 0 && pp[1] == 0 && pp[2] == 0) continue;
                double lon = std::atan2(pp[0], pp[2]);
                double lat = std::atan2(pp[1], std::hypot(pp[0], pp[2]));
                int u = static_cast<int>(std::lround((lon / lonMax + 1.0) / 2.0 * (erpW - 1)));
                int vv = static_cast<int>(std::lround((1.0 - lat / latMax) / 2.0 * (erpH - 1)));
                if (u < 0 || u >= erpW || vv < 0 || vv >= erpH) continue;
                float cur = fused.at<float>(vv, u);
                if (cur == 0 || dp[x] < cur) {
                    fused.at<float>(vv, u) = dp[x];
                    bandId.at<int>(vv, u) = static_cast<int>(b);
                    srcPix.at<int>(vv, u) = y * BW + x;
                }
            }
        }
    }

    // Gap fill: distanceTransform treats src==0 as the source, so with the
    // hole mask as input every hole gets the distance to (and 1-based pixel
    // index of) its nearest valid pixel.
    cv::Mat holes = (fused == 0);
    int holeCount = cv::countNonZero(holes);
    if (holeCount > 0 && holeCount < erpW * erpH) {
        cv::Mat dist, labels;
        cv::distanceTransform(holes, dist, labels, cv::DIST_L2, 5,
                              cv::DIST_LABEL_PIXEL);
        for (int y = 0; y < erpH; ++y) {
            const float* dp = dist.ptr<float>(y);
            const int* lp = labels.ptr<int>(y);
            float* fp = fused.ptr<float>(y);
            int* bp = bandId.ptr<int>(y);
            int* sp = srcPix.ptr<int>(y);
            for (int x = 0; x < erpW; ++x) {
                if (fp[x] == 0 && dp[x] > 0 && dp[x] <= fillGap && lp[x] > 0) {
                    int src = lp[x] - 1;   // nearest valid pixel index
                    int sy = src / erpW, sx = src % erpW;
                    fp[x] = fused.at<float>(sy, sx);
                    bp[x] = bandId.at<int>(sy, sx);
                    sp[x] = srcPix.at<int>(sy, sx);
                }
            }
        }
    }
    cv::medianBlur(fused, fused, 3);
}

// ---------------------------------------------------------------------------
// Point clouds
// ---------------------------------------------------------------------------

/// Push one point into the output buffers (caps at maxPoints).
struct PcOut {
    std::vector<float>& xyz;
    std::vector<uint8_t>& rgb;
    int maxPoints;
    PcOut(std::vector<float>& x, std::vector<uint8_t>& c, int max)
        : xyz(x), rgb(c), maxPoints(max) {}
    void push(double x, double y, double z, const cv::Vec3b& bgr)
    {
        if (static_cast<int>(xyz.size() / 3) >= maxPoints) return;
        xyz.push_back(static_cast<float>(x));
        xyz.push_back(static_cast<float>(y));
        xyz.push_back(static_cast<float>(z));
        rgb.push_back(bgr[2]);   // BGR -> RGB
        rgb.push_back(bgr[1]);
        rgb.push_back(bgr[0]);
    }
};

int pcStepFor(int N, int maxPoints)
{
    return std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N) / maxPoints))));
}

/// V1/V4/V5: rectified-frame points from depth + pinhole intrinsics.
void pcFromPinhole(const cv::Mat& depth, double fx, double fy, double cx, double cy,
                   const cv::Mat& rectColor, PcOut& out)
{
    const int H = depth.rows, W = depth.cols;
    const int step = pcStepFor(H * W, out.maxPoints);
    for (int y = 0; y < H; y += step) {
        const float* dp = depth.ptr<float>(y);
        const cv::Vec3b* cp = rectColor.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; x += step) {
            float z = dp[x];
            if (z <= 0) continue;
            out.push((x - cx) * z / fx, (y - cy) * z / fy, z, cp[x]);
        }
    }
}

/// V2: points from the triangulated (N,3) rectified-frame array.
void pcFromTriangulated(const cv::Mat& depth, const cv::Mat& pts,
                        const cv::Mat& rectColor, PcOut& out)
{
    const int H = depth.rows, W = depth.cols;
    const int step = pcStepFor(H * W, out.maxPoints);
    for (int y = 0; y < H; y += step) {
        const float* dp = depth.ptr<float>(y);
        const cv::Vec3b* cp = rectColor.ptr<cv::Vec3b>(y);
        for (int x = 0; x < W; x += step) {
            if (dp[x] <= 0) continue;
            const float* pp = pts.ptr<float>(y * W + x);
            out.push(pp[0], pp[1], pp[2], cp[x]);
        }
    }
}

/// V3: points from the fused ERP (ray direction from lon/lat * depth), colored
/// from the winning band's rectified image.
void pcFromErp(const cv::Mat& fused, const cv::Mat& bandId, const cv::Mat& srcPix,
               const std::vector<BandRes>& bands, double lonDeg, double latDeg, PcOut& out)
{
    const int H = fused.rows, W = fused.cols;
    const int step = pcStepFor(H * W, out.maxPoints);
    const double lonMax = deg2rad(lonDeg / 2.0);
    const double latMax = deg2rad(latDeg / 2.0);
    for (int y = 0; y < H; y += step) {
        const float* dp = fused.ptr<float>(y);
        const int* bp = bandId.ptr<int>(y);
        const int* sp = srcPix.ptr<int>(y);
        for (int x = 0; x < W; x += step) {
            float d = dp[x];
            if (d <= 0 || bp[x] < 0 || sp[x] < 0) continue;
            double lon = (static_cast<double>(x) / (W - 1) * 2.0 - 1.0) * lonMax;
            double lat = (1.0 - static_cast<double>(y) / (H - 1) * 2.0) * latMax;
            double cl = std::cos(lat);
            cv::Vec3b bgr(0, 0, 0);
            const cv::Mat& r = bands[bp[x]].rectL;
            int by = sp[x] / r.cols, bx = sp[x] % r.cols;
            if (by < r.rows && bx < r.cols) bgr = r.at<cv::Vec3b>(by, bx);
            out.push(cl * std::sin(lon) * d, std::sin(lat) * d, cl * std::cos(lon) * d, bgr);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Calibration loading
// ---------------------------------------------------------------------------

bool loadKalibr(const std::string& path, StereoCalib& out)
{
    std::ifstream f(path);
    if (!f.is_open()) return false;

    StereoCalib c;
    int section = -1;          // 0 = cam0, 1 = cam1
    bool inT = false;
    int tRow = 0;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t;
    bool haveT = false, haveRes = false;

    std::string line;
    while (std::getline(f, line)) {
        std::string s = trimLine(line);
        if (s == "cam0:") { section = 0; inT = false; continue; }
        if (s == "cam1:") { section = 1; inT = false; continue; }
        if (section < 0) continue;

        if (s.rfind("intrinsics:", 0) == 0) {
            std::vector<double> v = parseList(s);
            if (v.size() >= 4) {
                c.K[section] = cv::Matx33d(v[0], 0, v[2], 0, v[1], v[3], 0, 0, 1);
            }
        } else if (s.rfind("distortion_coeffs:", 0) == 0) {
            std::vector<double> v = parseList(s);
            for (size_t i = 0; i < v.size() && i < 4; ++i) c.D[section][i] = v[i];
        } else if (s.rfind("resolution:", 0) == 0) {
            std::vector<double> v = parseList(s);
            if (v.size() >= 2) {
                c.calibW = static_cast<int>(v[0]);
                c.calibH = static_cast<int>(v[1]);
                haveRes = true;
            }
        } else if (s.rfind("T_cn_cnm1:", 0) == 0) {
            inT = true; tRow = 0;
        } else if (inT && s.rfind("- [", 0) == 0) {
            std::vector<double> v = parseList(s);
            if (v.size() >= 4 && tRow < 3) {
                for (int col = 0; col < 3; ++col) R(tRow, col) = v[col];
                t[tRow] = v[3];
                ++tRow;
                if (tRow == 3) { inT = false; haveT = true; }
            }
        }
    }
    if (!haveT || !haveRes) return false;

    c.R = R;
    c.t = t;
    c.frameW = c.calibW;
    c.frameH = c.calibH;

    // Rectified frame: x along the baseline, z as close as possible to cam0's z
    cv::Vec3d t_w1 = -(R.t() * t);          // cam1 origin in cam0 frame
    c.baseline = cv::norm(t_w1);
    if (c.baseline < 1e-9) return false;
    cv::Vec3d xr = t_w1 / c.baseline;
    cv::Vec3d z0(0, 0, 1);
    cv::Vec3d zr = z0 - xr.dot(z0) * xr;
    zr /= cv::norm(zr);
    cv::Vec3d yr = zr.cross(xr);
    c.R_rect = cv::Matx33d(xr[0], xr[1], xr[2],
                           yr[0], yr[1], yr[2],
                           zr[0], zr[1], zr[2]);
    c.R1_rect = c.R_rect * R.t();           // rectified <- cam1
    c.t_rect = c.R_rect * t_w1;             // cam1 origin in rectified frame

    // Max off-axis angle covered by the sensor (image corners of cam0 + 2 deg)
    double maxTh = 0.0;
    const int corners[4][2] = {{0, 0}, {c.calibW - 1, 0},
                               {0, c.calibH - 1}, {c.calibW - 1, c.calibH - 1}};
    for (const auto& cc : corners) {
        cv::Vec3d ray = kb4BackProject(cc[0], cc[1], c.K[0], c.D[0]);
        maxTh = std::max(maxTh, std::acos(std::min(std::max(ray[2], -1.0), 1.0)));
    }
    c.lensMaxTheta = maxTh + deg2rad(2.0);

    out = c;
    return true;
}

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

bool CpuDepthProcessor::loadCalib(const std::string& calibPath)
{
    if (!cpu_depth::loadKalibr(calibPath, mCalib)) {
        mReady = false;
        return false;
    }
    mKCalib[0] = mCalib.K[0];
    mKCalib[1] = mCalib.K[1];
    mMaps.clear();
    mReady = true;
    return true;
}

void CpuDepthProcessor::setFrameSize(int w, int h)
{
    if (!mReady || w <= 0 || h <= 0) return;
    if (w == mCalib.frameW && h == mCalib.frameH) return;
    const double sx = static_cast<double>(w) / mCalib.calibW;
    const double sy = static_cast<double>(h) / mCalib.calibH;
    for (int c = 0; c < 2; ++c) {
        mCalib.K[c] = cv::Matx33d(mKCalib[c](0, 0) * sx, 0, mKCalib[c](0, 2) * sx,
                                  0, mKCalib[c](1, 1) * sy, mKCalib[c](1, 2) * sy,
                                  0, 0, 1);
    }
    mCalib.frameW = w;
    mCalib.frameH = h;
    mMaps.clear();
}

bool CpuDepthProcessor::buildMaps(const StereoCalib& c)
{
    mMaps.clear();
    const int W = c.frameW, H = c.frameH;

    if (mOpts.variant == V1_PINHOLE_SGBM || mOpts.variant == V4_PINHOLE_BM) {
        // V1/V4: fisheye -> pinhole rectification (same path as the SDK engine)
        cv::Mat K0(c.K[0]), D0(1, 4, CV_64F), K1(c.K[1]), D1(1, 4, CV_64F);
        for (int i = 0; i < 4; ++i) {
            D0.at<double>(i) = c.D[0][i];
            D1.at<double>(i) = c.D[1][i];
        }
        cv::Mat R1, R2, P1, P2, Q;
        cv::fisheye::stereoRectify(
            K0, D0, K1, D1, cv::Size(c.calibW, c.calibH), c.R, c.t,
            R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY,
            cv::Size(W, H), mOpts.balance, mOpts.fovScale);

        CpuMaps m;
        m.kind = 0;
        cv::fisheye::initUndistortRectifyMap(K0, D0, R1, P1, cv::Size(W, H),
                                             CV_32FC1, m.mapx0, m.mapy0);
        cv::fisheye::initUndistortRectifyMap(K1, D1, R2, P2, cv::Size(W, H),
                                             CV_32FC1, m.mapx1, m.mapy1);
        m.valid0 = (m.mapx0 >= 0);
        m.valid1 = (m.mapx1 >= 0);
        m.fx = P2.at<double>(0, 0);
        m.fy = P2.at<double>(1, 1);
        m.cx = P2.at<double>(0, 2);
        m.cy = P2.at<double>(1, 2);
        m.baseline = -P2.at<double>(0, 3) / m.fx;
        mMaps.push_back(m);
        return true;
    }

    if (mOpts.variant == V2V_ERP_SGBM || mOpts.variant == V2B_ERP_SGBM) {
        // V2: spherical rectification on a shared ray grid
        cv::Mat u, v;
        buildGrid(W, H, u, v);
        cv::Mat rays(H * W, 3, CV_32F);
        const double lonMax = deg2rad(mOpts.lonDeg);
        const double latMax = deg2rad(mOpts.latDeg);
        for (int y = 0; y < H; ++y) {
            const float* up = u.ptr<float>(y);
            const float* vp = v.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                float* r = rays.ptr<float>(y * W + x);
                if (mOpts.variant == V2V_ERP_SGBM) {
                    double lon = (up[x] - (W - 1) / 2.0) / std::max(W - 1, 1) * lonMax;
                    double lat = ((H - 1) / 2.0 - vp[x]) / std::max(H - 1, 1) * latMax;
                    double cl = std::cos(lat);
                    r[0] = static_cast<float>(cl * std::sin(lon));
                    r[1] = static_cast<float>(std::sin(lat));
                    r[2] = static_cast<float>(cl * std::cos(lon));
                } else {
                    // chi decreases with u so front points get positive disparity
                    double chi = kPi / 2.0 - (up[x] - (W - 1) / 2.0) / std::max(W - 1, 1) * lonMax;
                    double psi = (vp[x] - (H - 1) / 2.0) / std::max(H - 1, 1) * latMax + kPi / 2.0;
                    r[0] = static_cast<float>(std::cos(chi));
                    r[1] = static_cast<float>(std::sin(chi) * std::cos(psi));
                    r[2] = static_cast<float>(std::sin(chi) * std::sin(psi));
                }
            }
        }
        CpuMaps m;
        m.kind = 1;
        raysToRemapPair(rays, c, W, H, m.mapx0, m.mapy0, m.mapx1, m.mapy1,
                        m.valid0, m.valid1);
        m.raysRect = rays;
        m.tRect = c.t_rect;
        mMaps.push_back(m);
        return true;
    }

    if (mOpts.variant == V3_BANDS_SGBM) {
        // V3: one pinhole band per yaw angle; all bands share the ray grid so
        // rows are exactly epipolar
        const int BW = std::max(32, (static_cast<int>(W * mOpts.v3OutScale) / 16) * 16);
        const int BH = std::max(32, (static_cast<int>(H * mOpts.v3OutScale) / 16) * 16);
        const double f_v = (BW / 2.0) / std::tan(deg2rad(mOpts.bandHalfDeg));
        const double cy = (BH - 1) / 2.0;
        cv::Mat u, v;
        buildGrid(BW, BH, u, v);
        for (int b = 0; b < mOpts.nYaws; ++b) {
            const double yaw = deg2rad(mOpts.yaws[b]);
            const double cx = (BW - 1) / 2.0 - f_v * std::tan(yaw);
            cv::Mat rays(BH * BW, 3, CV_32F);
            for (int y = 0; y < BH; ++y) {
                const float* up = u.ptr<float>(y);
                const float* vp = v.ptr<float>(y);
                for (int x = 0; x < BW; ++x) {
                    double dx = (up[x] - cx) / f_v;
                    double dy = (vp[x] - cy) / f_v;
                    double n = std::sqrt(dx * dx + dy * dy + 1.0);
                    float* r = rays.ptr<float>(y * BW + x);
                    r[0] = static_cast<float>(dx / n);
                    r[1] = static_cast<float>(dy / n);
                    r[2] = static_cast<float>(1.0 / n);
                }
            }
            CpuMaps m;
            m.kind = 2;
            raysToRemapPair(rays, c, BW, BH, m.mapx0, m.mapy0, m.mapx1, m.mapy1,
                            m.valid0, m.valid1);
            m.raysRect = rays;
            m.tRect = c.t_rect;
            m.yaw = mOpts.yaws[b];
            m.f_v = f_v;
            mMaps.push_back(m);
        }
        return true;
    }

    return false;   // V5 has no CPU maps
}

bool CpuDepthProcessor::matchV3(const cv::Mat& left, const cv::Mat& right, CpuResult* out)
{
    std::vector<BandRes> bands;
    for (const CpuMaps& m : mMaps) {
        BandRes b;
        remapPair(left, right, m.mapx0, m.mapy0, m.mapx1, m.mapy1, b.rectL, b.rectR);
        cv::Mat d = sgbmMatch(b.rectL, b.rectR, mOpts, mOpts.v3Relaxed);
        triangulate(m.raysRect, d, m.tRect, mOpts.maxZ, 0.5, b.depth, b.pts);
        b.valid = m.valid0 & m.valid1;
        bands.push_back(b);
    }

    // ERP range matched to the band union; grid at ~4 px/deg (native density)
    const BandRes& b0 = bands.front();
    const CpuMaps& m0 = mMaps.front();
    const int BW = b0.depth.cols, BH = b0.depth.rows;
    const double vHalf = rad2deg(std::atan((BH / 2.0) / m0.f_v));
    const double hHalf = mOpts.bandHalfDeg;
    double maxAbsYaw = 0.0;
    for (int i = 0; i < mOpts.nYaws; ++i) maxAbsYaw = std::max(maxAbsYaw, std::abs(mOpts.yaws[i]));
    const double latDeg = 2.0 * (vHalf + 3.0);
    const double lonDeg = 2.0 * (maxAbsYaw + hHalf + 3.0);
    const int erpW = std::max(16, static_cast<int>(std::ceil(mOpts.erpDensity * lonDeg / 16.0)) * 16);
    const int erpH = std::max(16, static_cast<int>(std::ceil(mOpts.erpDensity * latDeg / 16.0)) * 16);

    cv::Mat fused, bandId, srcPix;
    fuseToErp(bands, erpW, erpH, lonDeg, latDeg, mOpts.erpFillGap, fused, bandId, srcPix);

    // Rectified display: the three band windows side by side
    cv::Mat stackedBands;
    for (const BandRes& b : bands) {
        cv::Mat one;
        cv::vconcat(b.rectL, b.rectR, one);
        if (stackedBands.empty()) stackedBands = one;
        else cv::hconcat(stackedBands, one, stackedBands);
    }

    if (out) {
        out->depth = fused;
        out->rectStereo = stackedBands;
        out->disparity = cv::Mat();
        out->pcXYZ.clear();
        out->pcRGB.clear();
        PcOut pc(out->pcXYZ, out->pcRGB, mOpts.maxPoints);
        pcFromErp(fused, bandId, srcPix, bands, lonDeg, latDeg, pc);
    }
    return true;
}

bool CpuDepthProcessor::process(const cv::Mat& stacked, const cv::Mat* engineRect,
                                const double* engineP0, double engineBaseline,
                                CpuResult* out)
{
    if (!mReady || stacked.rows < 2) return false;
    const int64 t0 = cv::getTickCount();

    const int H = stacked.rows / 2, W = stacked.cols;
    setFrameSize(W, H);

    const std::string key = [&]() {
        std::ostringstream ss;
        ss << mOpts.variant << '|' << W << 'x' << H << '|'
           << mOpts.balance << '|' << mOpts.fovScale << '|'
           << mOpts.lonDeg << '|' << mOpts.latDeg << '|'
           << mOpts.bandHalfDeg << '|' << mOpts.v3OutScale << '|' << mOpts.nYaws;
        for (int i = 0; i < mOpts.nYaws; ++i) ss << '|' << mOpts.yaws[i];
        return ss.str();
    }();
    if (mMapsKey != key) {
        mMapsKey = key;
        if (!buildMaps(mCalib)) return false;
    }

    const cv::Mat left = stacked.rowRange(0, H);
    const cv::Mat right = stacked.rowRange(H, 2 * H);

    CpuResult res;
    if (mOpts.variant == V5_ENGINE_SGBM) {
        if (!engineRect || engineRect->rows < 2) return false;
        const int EH = engineRect->rows / 2;
        cv::Mat eL = engineRect->rowRange(0, EH);
        cv::Mat eR = engineRect->rowRange(EH, 2 * EH);
        res.disparity = sgbmMatch(eL, eR, mOpts, false);
        const double fx = engineP0 ? engineP0[0] : 1.0;
        disparityToDepth(res.disparity, fx, engineBaseline, mOpts.maxZ, res.depth);
        res.rectStereo = engineRect->clone();
        res.pcXYZ.clear();
        res.pcRGB.clear();
        PcOut pc(res.pcXYZ, res.pcRGB, mOpts.maxPoints);
        const double fy = engineP0 ? engineP0[1] : fx;
        const double cx = engineP0 ? engineP0[2] : 0;
        const double cy = engineP0 ? engineP0[3] : 0;
        cv::Mat eL3;
        if (eL.channels() == 1) cv::cvtColor(eL, eL3, cv::COLOR_GRAY2BGR);
        else eL3 = eL;
        pcFromPinhole(res.depth, fx, fy, cx, cy, eL3, pc);
    } else if (mOpts.variant == V3_BANDS_SGBM) {
        if (!matchV3(left, right, &res)) return false;
    } else {
        const CpuMaps& m = mMaps[0];
        cv::Mat rL, rR;
        remapPair(left, right, m.mapx0, m.mapy0, m.mapx1, m.mapy1, rL, rR);
        cv::Mat rL3, rR3;
        if (rL.channels() == 1) cv::cvtColor(rL, rL3, cv::COLOR_GRAY2BGR); else rL3 = rL;
        if (rR.channels() == 1) cv::cvtColor(rR, rR3, cv::COLOR_GRAY2BGR); else rR3 = rR;
        cv::vconcat(rL3, rR3, res.rectStereo);

        if (mOpts.variant == V4_PINHOLE_BM) {
            res.disparity = bmMatch(rL, rR, mOpts);
        } else {
            res.disparity = sgbmMatch(rL, rR, mOpts, false);
        }
        res.pcXYZ.clear();
        res.pcRGB.clear();
        PcOut pc(res.pcXYZ, res.pcRGB, mOpts.maxPoints);
        if (m.kind == 0) {
            disparityToDepth(res.disparity, m.fx, m.baseline, mOpts.maxZ, res.depth);
            pcFromPinhole(res.depth, m.fx, m.fy, m.cx, m.cy, rL3, pc);
        } else {
            triangulate(m.raysRect, res.disparity, m.tRect, mOpts.maxZ, 0.5,
                        res.depth, res.pts);
            pcFromTriangulated(res.depth, res.pts, rL3, pc);
        }
    }

    mLastRuntimeMs = static_cast<double>(cv::getTickCount() - t0) * 1000.0
                   / cv::getTickFrequency();
    res.runtimeMs = mLastRuntimeMs;
    if (out) *out = std::move(res);
    return true;
}

} // namespace cpu_depth
