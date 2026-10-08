#include "detection_6d_foundationpose/foundationpose.hpp"
#include "tests/help_func.hpp"
#include "trt_core/trt_core.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <signal.h>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace detection_6d;
using namespace inference_core;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t kProtocolVersion = 1;
constexpr uint32_t kFlagInitialize = 1;
constexpr uint32_t kFlagStop = 2;
constexpr uint32_t kMaxWidth = 1920;
constexpr uint32_t kMaxHeight = 1080;

#pragma pack(push, 1)
struct FrameHeader
{
  char magic[4];
  uint32_t version;
  uint64_t frame_id;
  uint32_t width;
  uint32_t height;
  uint32_t flags;
  float K[9];
  uint64_t rgb_bytes;
  uint64_t depth_bytes;
  uint64_t mask_bytes;
};

struct PoseResponse
{
  char magic[4];
  uint32_t version;
  uint32_t status;
  uint64_t frame_id;
  float processing_ms;
  float pose[16];
};
#pragma pack(pop)

static_assert(sizeof(FrameHeader) == 88, "Unexpected frame header layout");
static_assert(sizeof(PoseResponse) == 88, "Unexpected pose response layout");

struct Options
{
  int port = 50051;
  int refine_iterations = 1;
  std::string refiner = "/workspace/models/refiner_hwc_dynamic_fp16.engine";
  std::string scorer = "/workspace/models/scorer_hwc_dynamic_fp16.engine";
  std::string mesh = "/workspace/test_data/apple/mesh.obj";
  std::string object = "apple";
  std::string output = "/workspace/test_output/live";
  std::string pointcloud_dir;
  std::string visualization_dir;
  int visualization_save_every = 0;
  int pointcloud_save_every = 0;
  int pointcloud_stride = 3;
  int pointcloud_max_points = 120000;
  int pointcloud_mesh_points = 12000;
  int pointcloud_reference_frames = 5;
  float pointcloud_min_depth = 0.15f;
  float pointcloud_max_depth = 2.0f;
  float pointcloud_crop_margin = 0.005f;
  float pointcloud_reference_voxel = 0.003f;
};

Options ParseArgs(int argc, char **argv)
{
  Options options;
  const std::unordered_map<std::string, std::string *> strings = {
      {"--refiner", &options.refiner}, {"--scorer", &options.scorer},
      {"--mesh", &options.mesh}, {"--object", &options.object}, {"--output", &options.output},
      {"--pointcloud-dir", &options.pointcloud_dir},
      {"--visualization-dir", &options.visualization_dir}};
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto it = strings.find(arg);
    if (it != strings.end() && i + 1 < argc)
      *it->second = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      options.port = std::stoi(argv[++i]);
    else if (arg == "--refine-iterations" && i + 1 < argc)
      options.refine_iterations = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-save-every" && i + 1 < argc)
      options.pointcloud_save_every = std::stoi(argv[++i]);
    else if (arg == "--visualization-save-every" && i + 1 < argc)
      options.visualization_save_every = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-stride" && i + 1 < argc)
      options.pointcloud_stride = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-max-points" && i + 1 < argc)
      options.pointcloud_max_points = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-mesh-points" && i + 1 < argc)
      options.pointcloud_mesh_points = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-reference-frames" && i + 1 < argc)
      options.pointcloud_reference_frames = std::stoi(argv[++i]);
    else if (arg == "--pointcloud-min-depth" && i + 1 < argc)
      options.pointcloud_min_depth = std::stof(argv[++i]);
    else if (arg == "--pointcloud-max-depth" && i + 1 < argc)
      options.pointcloud_max_depth = std::stof(argv[++i]);
    else if (arg == "--pointcloud-crop-margin" && i + 1 < argc)
      options.pointcloud_crop_margin = std::stof(argv[++i]);
    else if (arg == "--pointcloud-reference-voxel" && i + 1 < argc)
      options.pointcloud_reference_voxel = std::stof(argv[++i]);
    else
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
  }
  if (options.visualization_save_every < 0 || options.pointcloud_save_every < 0 ||
      options.pointcloud_stride < 1 ||
      options.pointcloud_max_points < 1 || options.pointcloud_mesh_points < 100 ||
      options.pointcloud_reference_frames < 1 || options.pointcloud_crop_margin < 0.0f ||
      options.pointcloud_reference_voxel <= 0.0f || options.pointcloud_min_depth < 0.0f ||
      options.pointcloud_max_depth <= options.pointcloud_min_depth)
    throw std::runtime_error("Invalid point-cloud options");
  return options;
}

bool ReceiveAll(int fd, void *buffer, size_t bytes)
{
  auto *cursor = static_cast<uint8_t *>(buffer);
  while (bytes > 0)
  {
    const ssize_t count = recv(fd, cursor, bytes, 0);
    if (count <= 0)
      return false;
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

bool SendAll(int fd, const void *buffer, size_t bytes)
{
  const auto *cursor = static_cast<const uint8_t *>(buffer);
  while (bytes > 0)
  {
    const ssize_t count = send(fd, cursor, bytes, MSG_NOSIGNAL);
    if (count <= 0)
      return false;
    cursor += count;
    bytes -= static_cast<size_t>(count);
  }
  return true;
}

void ValidateHeader(const FrameHeader &header, bool require_mask)
{
  if (std::memcmp(header.magic, "FPLV", 4) != 0 || header.version != kProtocolVersion)
    throw std::runtime_error("Invalid live-stream protocol header");
  if (header.width == 0 || header.height == 0 || header.width > kMaxWidth ||
      header.height > kMaxHeight)
    throw std::runtime_error("Invalid frame dimensions");
  const uint64_t pixels = static_cast<uint64_t>(header.width) * header.height;
  if (header.rgb_bytes != pixels * 3 || header.depth_bytes != pixels * 2 ||
      header.mask_bytes != (require_mask ? pixels : 0))
    throw std::runtime_error("Invalid frame payload sizes");
}

struct Frame
{
  FrameHeader header{};
  cv::Mat rgb;
  cv::Mat depth;
  cv::Mat mask;
};

bool ReceiveFrame(int fd, Frame &frame)
{
  if (!ReceiveAll(fd, &frame.header, sizeof(frame.header)))
    return false;
  if ((frame.header.flags & kFlagStop) != 0)
    return true;
  const bool initialize = (frame.header.flags & kFlagInitialize) != 0;
  ValidateHeader(frame.header, initialize);

  std::vector<uint8_t> rgb(frame.header.rgb_bytes);
  std::vector<uint16_t> depth(frame.header.depth_bytes / sizeof(uint16_t));
  std::vector<uint8_t> mask(frame.header.mask_bytes);
  if (!ReceiveAll(fd, rgb.data(), rgb.size()) ||
      !ReceiveAll(fd, depth.data(), frame.header.depth_bytes) ||
      (!mask.empty() && !ReceiveAll(fd, mask.data(), mask.size())))
    return false;

  cv::Mat bgr(frame.header.height, frame.header.width, CV_8UC3, rgb.data());
  cv::cvtColor(bgr, frame.rgb, cv::COLOR_BGR2RGB);
  cv::Mat depth_mm(frame.header.height, frame.header.width, CV_16UC1, depth.data());
  depth_mm.convertTo(frame.depth, CV_32FC1, 0.001);
  if (!mask.empty())
    frame.mask = cv::Mat(frame.header.height, frame.header.width, CV_8UC1, mask.data()).clone();
  return true;
}

bool SendPose(int fd, uint64_t frame_id, float processing_ms, const Eigen::Matrix4f &pose,
              uint32_t status = 0)
{
  PoseResponse response{};
  std::memcpy(response.magic, "FPOS", 4);
  response.version = kProtocolVersion;
  response.status = status;
  response.frame_id = frame_id;
  response.processing_ms = processing_ms;
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col)
      response.pose[row * 4 + col] = pose(row, col);
  return SendAll(fd, &response, sizeof(response));
}

void SavePose(const fs::path &path, const Eigen::Matrix4f &pose)
{
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream stream(temporary);
  stream << std::setprecision(9) << pose << std::endl;
  stream.close();
  fs::rename(temporary, path);
}

#pragma pack(push, 1)
struct PlyVertex
{
  float x;
  float y;
  float z;
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};
#pragma pack(pop)

using PointCloud = std::vector<PlyVertex>;

size_t SavePointCloud(const fs::path &path, const PointCloud &vertices, int max_points)
{
  const size_t output_count = std::min(vertices.size(), static_cast<size_t>(max_points));
  fs::create_directories(path.parent_path());
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream stream(temporary, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot write point cloud: " + temporary.string());
  stream << "ply\nformat binary_little_endian 1.0\nelement vertex " << output_count
         << "\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  for (size_t i = 0; i < output_count; ++i)
  {
    const size_t source = output_count == vertices.size() ? i :
        i * (vertices.size() - 1) / std::max<size_t>(output_count - 1, 1);
    stream.write(reinterpret_cast<const char *>(&vertices[source]), sizeof(PlyVertex));
  }
  stream.close();
  fs::rename(temporary, path);
  return output_count;
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> MeshBounds(
    const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  const auto &mesh_vertices = mesh_loader->GetMeshVertices();
  if (mesh_vertices.empty())
    throw std::runtime_error("Cannot create target point cloud from an empty mesh");
  Eigen::Vector3f low = mesh_vertices.front();
  Eigen::Vector3f high = mesh_vertices.front();
  for (const auto &point : mesh_vertices)
  {
    low = low.cwiseMin(point);
    high = high.cwiseMax(point);
  }
  return {low, high};
}

PointCloud BuildObservedCloud(const Frame &frame, const Eigen::Matrix4f &pose,
                              const Eigen::Vector3f &mesh_low,
                              const Eigen::Vector3f &mesh_high, const Options &options,
                              bool use_mask)
{
  const float fx = frame.header.K[0];
  const float fy = frame.header.K[4];
  const float cx = frame.header.K[2];
  const float cy = frame.header.K[5];
  if (fx <= 0.0f || fy <= 0.0f)
    throw std::runtime_error("Cannot generate point cloud with invalid intrinsics");

  PointCloud vertices;
  vertices.reserve(static_cast<size_t>(frame.depth.total()) /
                   (options.pointcloud_stride * options.pointcloud_stride));
  for (int v = 0; v < frame.depth.rows; v += options.pointcloud_stride)
  {
    const float *depth_row = frame.depth.ptr<float>(v);
    const cv::Vec3b *rgb_row = frame.rgb.ptr<cv::Vec3b>(v);
    for (int u = 0; u < frame.depth.cols; u += options.pointcloud_stride)
    {
      const float z = depth_row[u];
      if (!std::isfinite(z) || z < options.pointcloud_min_depth ||
          z > options.pointcloud_max_depth)
        continue;
      if (use_mask && (frame.mask.empty() || frame.mask.at<uint8_t>(v, u) == 0))
        continue;
      const Eigen::Vector3f camera_point((u - cx) * z / fx, (v - cy) * z / fy, z);
      const Eigen::Vector3f model_point =
          pose.block<3, 3>(0, 0).transpose() * (camera_point - pose.block<3, 1>(0, 3));
      const Eigen::Vector3f margin =
          Eigen::Vector3f::Constant(options.pointcloud_crop_margin);
      if ((model_point.array() < (mesh_low - margin).array()).any() ||
          (model_point.array() > (mesh_high + margin).array()).any())
        continue;
      const cv::Vec3b color = rgb_row[u];
      vertices.push_back({camera_point.x(), camera_point.y(), camera_point.z(),
                          color[0], color[1], color[2]});
    }
  }
  return vertices;
}

PointCloud SampleMeshCloud(const std::shared_ptr<BaseMeshLoader> &mesh_loader, int count)
{
  const auto &vertices = mesh_loader->GetMeshVertices();
  const auto &faces = mesh_loader->GetMeshTriangleFaces();
  PointCloud cloud;
  cloud.reserve(count);
  if (faces.empty())
  {
    for (int i = 0; i < count; ++i)
    {
      const Eigen::Vector3f &point = vertices[static_cast<size_t>(i) % vertices.size()];
      cloud.push_back({point.x(), point.y(), point.z(), 255, 65, 50});
    }
    return cloud;
  }

  std::vector<double> cumulative;
  cumulative.reserve(faces.size());
  double total_area = 0.0;
  for (const auto &face : faces)
  {
    total_area += 0.5 * static_cast<double>(
        (vertices[face[1]] - vertices[face[0]]).cross(
            vertices[face[2]] - vertices[face[0]]).norm());
    cumulative.push_back(total_area);
  }
  if (total_area <= 0.0)
    throw std::runtime_error("Cannot sample mesh point cloud from zero-area faces");

  std::mt19937 generator(0x46504c56);
  std::uniform_real_distribution<double> area_distribution(0.0, total_area);
  std::uniform_real_distribution<float> unit_distribution(0.0f, 1.0f);
  for (int i = 0; i < count; ++i)
  {
    const size_t face_index = static_cast<size_t>(std::lower_bound(
        cumulative.begin(), cumulative.end(), area_distribution(generator)) - cumulative.begin());
    const auto &face = faces[std::min(face_index, faces.size() - 1)];
    float a = unit_distribution(generator);
    float b = unit_distribution(generator);
    if (a + b > 1.0f)
    {
      a = 1.0f - a;
      b = 1.0f - b;
    }
    const Eigen::Vector3f point = vertices[face[0]] +
        a * (vertices[face[1]] - vertices[face[0]]) +
        b * (vertices[face[2]] - vertices[face[0]]);
    cloud.push_back({point.x(), point.y(), point.z(), 255, 65, 50});
  }
  return cloud;
}

PointCloud TransformCloud(const PointCloud &model_cloud, const Eigen::Matrix4f &pose,
                          uint8_t red, uint8_t green, uint8_t blue)
{
  PointCloud camera_cloud;
  camera_cloud.reserve(model_cloud.size());
  for (const auto &sample : model_cloud)
  {
    const Eigen::Vector3f point = pose.block<3, 3>(0, 0) *
        Eigen::Vector3f(sample.x, sample.y, sample.z) + pose.block<3, 1>(0, 3);
    camera_cloud.push_back({point.x(), point.y(), point.z(), red, green, blue});
  }
  return camera_cloud;
}

class ReferenceCloud
{
public:
  ReferenceCloud(int frames, float voxel) : target_frames_(frames), voxel_(voxel) {}

  void Add(const PointCloud &camera_cloud, const Eigen::Matrix4f &pose)
  {
    if (ready_ || camera_cloud.empty())
      return;
    const Eigen::Matrix3f rotation_inverse = pose.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3f translation = pose.block<3, 1>(0, 3);
    for (const auto &sample : camera_cloud)
    {
      const Eigen::Vector3f point = rotation_inverse *
          (Eigen::Vector3f(sample.x, sample.y, sample.z) - translation);
      samples_.push_back({point.x(), point.y(), point.z(), sample.red, sample.green, sample.blue});
    }
    if (++frames_ >= target_frames_)
      Fuse();
  }

  PointCloud AtPose(const Eigen::Matrix4f &pose) const
  {
    return TransformCloud(samples_, pose, 255, 215, 35);
  }

  bool Ready() const { return ready_; }

private:
  void Fuse()
  {
    struct Sum
    {
      Eigen::Vector3d point = Eigen::Vector3d::Zero();
      size_t count = 0;
    };
    std::map<std::tuple<int, int, int>, Sum> voxels;
    for (const auto &sample : samples_)
    {
      const auto key = std::make_tuple(
          static_cast<int>(std::floor(sample.x / voxel_)),
          static_cast<int>(std::floor(sample.y / voxel_)),
          static_cast<int>(std::floor(sample.z / voxel_)));
      auto &sum = voxels[key];
      sum.point += Eigen::Vector3d(sample.x, sample.y, sample.z);
      ++sum.count;
    }
    samples_.clear();
    samples_.reserve(voxels.size());
    for (const auto &[key, sum] : voxels)
    {
      (void)key;
      const Eigen::Vector3d point = sum.point / static_cast<double>(sum.count);
      samples_.push_back({static_cast<float>(point.x()), static_cast<float>(point.y()),
                          static_cast<float>(point.z()), 255, 215, 35});
    }
    ready_ = true;
  }

  int target_frames_;
  int frames_ = 0;
  float voxel_;
  bool ready_ = false;
  PointCloud samples_;
};

void SaveTargetClouds(const fs::path &root, uint64_t frame_id,
                      const PointCloud &observed, const PointCloud &mesh,
                      const PointCloud &reference, const Options &options)
{
  std::ostringstream name;
  name << std::setw(6) << std::setfill('0') << frame_id << ".ply";
  const size_t observed_count = SavePointCloud(root / "observed" / name.str(), observed,
                                                options.pointcloud_max_points);
  const size_t mesh_count = SavePointCloud(root / "mesh" / name.str(), mesh,
                                            options.pointcloud_mesh_points);
  const size_t reference_count = SavePointCloud(root / "reference" / name.str(), reference,
                                                 options.pointcloud_max_points);
  std::cout << "POINTCLOUD frame=" << frame_id << " observed=" << observed_count
            << " mesh=" << mesh_count << " reference=" << reference_count << std::endl;
}

void SaveVisualization(const fs::path &path, const Frame &frame, const Eigen::Matrix3f &K,
                       const Eigen::Matrix4f &pose,
                       const std::shared_ptr<BaseMeshLoader> &mesh_loader)
{
  fs::create_directories(path.parent_path());
  cv::Mat visualization;
  cv::cvtColor(frame.rgb, visualization, cv::COLOR_RGB2BGR);
  draw3DBoundingBox(K, ConvertPoseMesh2BBox(pose, mesh_loader), visualization.rows,
                    visualization.cols, mesh_loader->GetObjectDimension(), visualization);
  if (!cv::imwrite(path.string(), visualization))
    throw std::runtime_error("Cannot write visualization: " + path.string());
}

int CreateServer(int port)
{
  const int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0)
    throw std::runtime_error("socket() failed");
  int reuse = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      listen(server, 2) != 0)
  {
    close(server);
    throw std::runtime_error("Failed to bind live server port " + std::to_string(port));
  }
  return server;
}

} // namespace

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  signal(SIGPIPE, SIG_IGN);
  const Options options = ParseArgs(argc, argv);
  fs::create_directories(options.output);

  const int server = CreateServer(options.port);
  std::cout << "READY port=" << options.port << std::endl;

  int client = -1;
  Frame first;
  while (true)
  {
    client = accept(server, nullptr, nullptr);
    if (client < 0)
      continue;
    int no_delay = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    if (ReceiveFrame(client, first) && (first.header.flags & kFlagInitialize) != 0)
      break;
    close(client);
    client = -1;
  }
  close(server);

  Eigen::Matrix3f K;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      K(row, col) = first.header.K[row * 3 + col];

  auto refiner_core = CreateTrtInferCore(
      options.refiner,
      {{"transf_input", {252, 160, 160, 6}}, {"render_input", {252, 160, 160, 6}}},
      {{"trans", {252, 3}}, {"rot", {252, 3}}}, 1);
  auto scorer_core = CreateTrtInferCore(
      options.scorer,
      {{"transf_input", {252, 160, 160, 6}}, {"render_input", {252, 160, 160, 6}}},
      {{"scores", {252, 1}}}, 1);
  auto mesh_loader = CreateAssimpMeshLoader(options.object, options.mesh);
  const auto [mesh_low, mesh_high] = MeshBounds(mesh_loader);
  const PointCloud mesh_model_cloud = SampleMeshCloud(mesh_loader, options.pointcloud_mesh_points);
  ReferenceCloud reference_cloud(options.pointcloud_reference_frames,
                                 options.pointcloud_reference_voxel);
  auto foundation_pose = CreateFoundationPoseModel(
      refiner_core, scorer_core, {mesh_loader}, K, first.rgb.rows, first.rgb.cols);

  Eigen::Matrix4f pose;
  auto start = std::chrono::steady_clock::now();
  if (!foundation_pose->Register(first.rgb, first.depth, first.mask, options.object, pose,
                                 options.refine_iterations))
  {
    const Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    SendPose(client, first.header.frame_id, 0.0f, identity, 1);
    std::cerr << "SERVICE_ERROR registration failed" << std::endl;
    close(client);
    return 1;
  }
  auto end = std::chrono::steady_clock::now();
  float elapsed = std::chrono::duration<float, std::milli>(end - start).count();
  SavePose(fs::path(options.output) / "latest_pose.txt", pose);
  if (!options.visualization_dir.empty() && options.visualization_save_every > 0)
    SaveVisualization(fs::path(options.visualization_dir) / "000000.png", first, K, pose,
                      mesh_loader);
  if (!options.pointcloud_dir.empty() && options.pointcloud_save_every > 0)
  {
    const PointCloud observed = BuildObservedCloud(first, pose, mesh_low, mesh_high, options, true);
    reference_cloud.Add(observed, pose);
    SaveTargetClouds(options.pointcloud_dir, 0, observed,
                     TransformCloud(mesh_model_cloud, pose, 255, 65, 50),
                     reference_cloud.AtPose(pose), options);
  }
  if (!SendPose(client, first.header.frame_id, elapsed, pose))
    return 1;
  std::cout << "REGISTER frame=" << first.header.frame_id << " ms=" << elapsed << std::endl;

  while (true)
  {
    Frame frame;
    if (!ReceiveFrame(client, frame) || (frame.header.flags & kFlagStop) != 0)
      break;
    if (frame.header.width != first.header.width || frame.header.height != first.header.height)
      throw std::runtime_error("Live frame dimensions changed");

    if ((frame.header.flags & kFlagInitialize) != 0)
    {
      Eigen::Matrix4f registered_pose;
      start = std::chrono::steady_clock::now();
      if (!foundation_pose->Register(frame.rgb, frame.depth, frame.mask, options.object,
                                     registered_pose, options.refine_iterations))
      {
        SendPose(client, frame.header.frame_id, 0.0f, pose, 4);
        std::cerr << "REREGISTER_REJECTED frame=" << frame.header.frame_id << std::endl;
        continue;
      }
      end = std::chrono::steady_clock::now();
      elapsed = std::chrono::duration<float, std::milli>(end - start).count();
      pose = registered_pose;
      SavePose(fs::path(options.output) / "latest_pose.txt", pose);
      if (!SendPose(client, frame.header.frame_id, elapsed, pose))
        break;
      std::cout << "REREGISTER frame=" << frame.header.frame_id << " ms=" << elapsed
                << std::endl;
      continue;
    }

    Eigen::Matrix4f tracked_pose;
    start = std::chrono::steady_clock::now();
    if (!foundation_pose->Track(frame.rgb, frame.depth, pose, options.object, tracked_pose,
                                options.refine_iterations))
    {
      SendPose(client, frame.header.frame_id, 0.0f, pose, 2);
      std::cerr << "SERVICE_ERROR tracking failed at frame=" << frame.header.frame_id
                << std::endl;
      break;
    }
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration<float, std::milli>(end - start).count();
    pose = tracked_pose;
    PointCloud observed;
    if (!options.pointcloud_dir.empty() && options.pointcloud_save_every > 0 &&
        (!reference_cloud.Ready() ||
         frame.header.frame_id % options.pointcloud_save_every == 0))
    {
      observed = BuildObservedCloud(frame, pose, mesh_low, mesh_high, options, false);
      reference_cloud.Add(observed, pose);
    }
    SavePose(fs::path(options.output) / "latest_pose.txt", pose);
    if (!options.visualization_dir.empty() && options.visualization_save_every > 0 &&
        frame.header.frame_id % options.visualization_save_every == 0)
    {
      std::ostringstream name;
      name << std::setw(6) << std::setfill('0') << frame.header.frame_id << ".png";
      SaveVisualization(fs::path(options.visualization_dir) / name.str(), frame, K, pose,
                        mesh_loader);
    }
    if (!options.pointcloud_dir.empty() && options.pointcloud_save_every > 0 &&
        frame.header.frame_id % options.pointcloud_save_every == 0)
    {
      if (observed.empty())
        observed = BuildObservedCloud(frame, pose, mesh_low, mesh_high, options, false);
      SaveTargetClouds(options.pointcloud_dir, frame.header.frame_id, observed,
                       TransformCloud(mesh_model_cloud, pose, 255, 65, 50),
                       reference_cloud.AtPose(pose), options);
    }
    if (!SendPose(client, frame.header.frame_id, elapsed, pose))
      break;
    if (frame.header.frame_id % 30 == 0)
      std::cout << "TRACK frame=" << frame.header.frame_id << " ms=" << elapsed << std::endl;
  }

  close(client);
  std::cout << "STOP" << std::endl;
  return 0;
}
