#include "tests/help_func.hpp"
#include "tests/fs_util.hpp"

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace detection_6d;
using namespace inference_core;
namespace fs = std::filesystem;

struct Options
{
  std::string dataset = "/workspace/test_data/apple";
  std::string refiner = "/workspace/models/refiner_hwc_dynamic_fp16.engine";
  std::string scorer  = "/workspace/models/scorer_hwc_dynamic_fp16.engine";
  std::string output  = "/workspace/test_output/apple";
  std::string object  = "apple";
  int limit = 10;
  int refine_iterations = 1;
};

static Options ParseArgs(int argc, char **argv)
{
  Options options;
  const std::unordered_map<std::string, std::string *> strings = {
      {"--dataset", &options.dataset}, {"--refiner", &options.refiner},
      {"--scorer", &options.scorer},   {"--output", &options.output},
      {"--object", &options.object}};
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto it = strings.find(arg);
    if (it != strings.end() && i + 1 < argc)
    {
      *it->second = argv[++i];
    }
    else if (arg == "--limit" && i + 1 < argc)
    {
      options.limit = std::stoi(argv[++i]);
    }
    else if (arg == "--refine-iterations" && i + 1 < argc)
    {
      options.refine_iterations = std::stoi(argv[++i]);
    }
    else
    {
      throw std::runtime_error("Unknown or incomplete argument: " + arg);
    }
  }
  return options;
}

static void SavePose(const fs::path &path, const Eigen::Matrix4f &pose)
{
  std::ofstream stream(path);
  stream << std::setprecision(9) << pose << std::endl;
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  const Options options = ParseArgs(argc, argv);
  const fs::path dataset(options.dataset);
  const fs::path output(options.output);
  fs::create_directories(output / "poses");
  fs::create_directories(output / "vis");

  auto rgb_paths = get_files_in_directory((dataset / "rgb").string());
  std::sort(rgb_paths.begin(), rgb_paths.end());
  if (rgb_paths.empty())
  {
    throw std::runtime_error("No RGB frames in " + (dataset / "rgb").string());
  }
  if (options.limit > 0 && rgb_paths.size() > static_cast<size_t>(options.limit))
  {
    rgb_paths.resize(options.limit);
  }

  const std::string first_id = rgb_paths.front().stem().string();
  auto [first_rgb, first_depth, first_mask] = ReadRgbDepthMask(
      rgb_paths.front().string(), (dataset / "depth" / (first_id + ".png")).string(),
      (dataset / "masks" / (first_id + ".png")).string());
  const Eigen::Matrix3f K = ReadCamK((dataset / "cam_K.txt").string());

  auto refiner_core = CreateTrtInferCore(
      options.refiner,
      {{"transf_input", {252, 160, 160, 6}}, {"render_input", {252, 160, 160, 6}}},
      {{"trans", {252, 3}}, {"rot", {252, 3}}}, 1);
  auto scorer_core = CreateTrtInferCore(
      options.scorer,
      {{"transf_input", {252, 160, 160, 6}}, {"render_input", {252, 160, 160, 6}}},
      {{"scores", {252, 1}}}, 1);
  auto mesh_loader = CreateAssimpMeshLoader(options.object, (dataset / "mesh.obj").string());
  auto foundation_pose = CreateFoundationPoseModel(
      refiner_core, scorer_core, {mesh_loader}, K, first_rgb.rows, first_rgb.cols);

  Eigen::Matrix4f pose;
  auto start = std::chrono::steady_clock::now();
  if (!foundation_pose->Register(first_rgb, first_depth, first_mask, options.object, pose,
                                 options.refine_iterations))
  {
    throw std::runtime_error("FoundationPose register failed");
  }
  auto end = std::chrono::steady_clock::now();
  const double register_ms = std::chrono::duration<double, std::milli>(end - start).count();
  SavePose(output / "poses" / (first_id + ".txt"), pose);

  cv::Mat first_vis = first_rgb.clone();
  cv::cvtColor(first_vis, first_vis, cv::COLOR_RGB2BGR);
  draw3DBoundingBox(K, ConvertPoseMesh2BBox(pose, mesh_loader), first_vis.rows, first_vis.cols,
                    mesh_loader->GetObjectDimension(), first_vis);
  cv::imwrite((output / "vis" / (first_id + ".png")).string(), first_vis);
  std::cout << "REGISTER frame=" << first_id << " ms=" << register_ms << std::endl
            << pose << std::endl;

  double track_total_ms = 0.0;
  for (size_t i = 1; i < rgb_paths.size(); ++i)
  {
    const std::string frame_id = rgb_paths[i].stem().string();
    auto [rgb, depth] = ReadRgbDepth(rgb_paths[i].string(),
                                     (dataset / "depth" / (frame_id + ".png")).string());
    Eigen::Matrix4f tracked_pose;
    start = std::chrono::steady_clock::now();
    if (!foundation_pose->Track(rgb, depth, pose, options.object, tracked_pose,
                                options.refine_iterations))
    {
      throw std::runtime_error("FoundationPose track failed at frame " + frame_id);
    }
    end = std::chrono::steady_clock::now();
    const double track_ms = std::chrono::duration<double, std::milli>(end - start).count();
    track_total_ms += track_ms;
    pose = tracked_pose;
    SavePose(output / "poses" / (frame_id + ".txt"), pose);

    cv::Mat vis = rgb.clone();
    cv::cvtColor(vis, vis, cv::COLOR_RGB2BGR);
    draw3DBoundingBox(K, ConvertPoseMesh2BBox(pose, mesh_loader), vis.rows, vis.cols,
                      mesh_loader->GetObjectDimension(), vis);
    cv::imwrite((output / "vis" / (frame_id + ".png")).string(), vis);
    std::cout << "TRACK frame=" << frame_id << " ms=" << track_ms << std::endl;
  }

  const size_t track_count = rgb_paths.size() - 1;
  std::cout << "SUMMARY frames=" << rgb_paths.size() << " register_ms=" << register_ms
            << " track_avg_ms=" << (track_count ? track_total_ms / track_count : 0.0)
            << std::endl;
  return 0;
}
