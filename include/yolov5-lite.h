#pragma once

#include <string>

#include "iyolo.h"
#include "rknn_api.h"
#include "utils/image_utils.h"

namespace rknn_yolo_inference {

class Yolov5Lite : public IYolo {
  typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;  // 是否为量化模型（性能需求比浮点模型更小）
  } rknn_app_context_t;

  int OBJ_NAME_MAX_SIZE = 64;
  static constexpr int OBJ_NUMB_MAX_SIZE = 128;
  int OBJ_CLASS_NUM = 82;
  float NMS_THRESH = 0.45;
  float BOX_THRESH = 0.25;
  int PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM;

  typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
  } object_detect_result;

  typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
  } object_detect_result_list;

 public:
  Yolov5Lite(std::string model, std::string label_path);

  DetectResult Detect(cv::Mat img0) override;

 private:
  /**
   * @brief 初始化和装载 rknn_app_context_t
   * @param model_path
   */
  void InitModel(std::string model_path);
  void ReleaseModel() {}
  void PostProcess();
  // 来自 model_zoo 例程
  int InitPostProcess(std::string label_path);
  void DeinitPostProcess();
  const char *ClsIdtoName(int cls_id);
  int PostProcess(rknn_app_context_t *app_ctx, void *outputs,
                  letterbox_t *letter_box, float conf_threshold,
                  float nms_threshold, object_detect_result_list *od_results);

  /**
   * @brief
   * 本模块的模型直接输出的是三个尺寸的特征图结果，所以还需要对输出进行比较繁琐的后处理，
   *  才能得到最直观的结果。以下部分是后处理的部分，主要由几个关键步骤组成：
   *  - 从输出端获取输出结果，此步会根据量化/非量化模型选择不同的处理方式
   *  — 读取特征图结果，整理为检测目标框、标签Id、置信度
   *  TODO: 该步骤注意程序和模型的标签种类数量是否一致，否则无法正常读取结果。
   *  - 进行非极大值抑制
   * 以下是一些常见的方法参数：
   * @param anchor 检测框锚点数组
   * @param grid_h 特征图的网格高度
   * @param grid_w 特征图的网格宽度
   * @param height 原始图像的高度
   * @param width 原始图像的宽度
   * @param stride 网格的步幅
   * @param boxes 输出检测框的坐标信息 (x, y, w, h)
   * @param objProbs 输出目标的置信度
   * @param classId 输出目标的类别 ID
   * @param threshold 检测框过滤的置信度阈值
   * @param zp 零点，用于量化反量化
   * @param scale 量化比例，用于反量化
   */

  /**
   * @brief 处理 int8 格式的输入数据，根据检测框架生成检测框信息
   *
   * @param input 输入的 int8 格式的特征图数据
   * @return int 返回状态码，0 表示成功
   */
  int ProcessInt8(int8_t *input, int *anchor, int grid_h, int grid_w,
                  int height, int width, int stride, std::vector<float> &boxes,
                  std::vector<float> &objProbs, std::vector<int> &classId,
                  float threshold, int32_t zp, float scale);

  /**
   * @brief 处理浮点型 (FP32) 输入数据，根据检测框架生成检测框信息
   *
   * @param input 输入的浮点型特征图数据
   * @return int 返回状态码，0 表示成功
   */
  int ProcessFP32(float *input, int *anchor, int grid_h, int grid_w, int height,
                  int width, int stride, std::vector<float> &boxes,
                  std::vector<float> &objProbs, std::vector<int> &classId,
                  float threshold);

  /**
   * @brief 处理 uint8 格式的输入数据，根据检测框架生成检测框信息
   *
   * @param input 输入的 uint8 格式的特征图数据
   * @return int 返回状态码，0 表示成功
   */
  int ProcessUInt8(uint8_t *input, int *anchor, int grid_h, int grid_w,
                   int height, int width, int stride, std::vector<float> &boxes,
                   std::vector<float> &objProbs, std::vector<int> &classId,
                   float threshold, int32_t zp, float scale);

  /**
   * @brief 裁剪浮点值至指定范围
   *
   * @param val 输入值
   * @param min 最小值
   * @param max 最大值
   * @return int32_t 返回裁剪后的值
   */
  inline static int32_t __clip(float val, float min, float max);

  /**
   * @brief 将 int8 量化值反量化为浮点数
   *
   * @param qnt 输入的量化值
   * @param zp 零点
   * @param scale 量化比例
   * @return float 返回反量化后的浮点值
   */
  static float DeqntAffinetoFP32(int8_t qnt, int32_t zp, float scale);

  /**
   * @brief 将 uint8 量化值反量化为浮点数
   *
   * @param qnt 输入的量化值
   * @param zp 零点
   * @param scale 量化比例
   * @return float 返回反量化后的浮点值
   */
  static float DeqntAffineUInt8toFP32(uint8_t qnt, int32_t zp, float scale);

  /**
   * @brief 将浮点值量化为 uint8 值
   *
   * @param f32 输入的浮点值
   * @param zp 零点
   * @param scale 量化比例
   * @return uint8_t 返回量化后的 uint8 值
   */
  static uint8_t QntFP32toAffineUInt8(float f32, int32_t zp, float scale);

  /**
   * @brief 将浮点值量化为 int8 值
   *
   * @param f32 输入的浮点值
   * @param zp 零点
   * @param scale 量化比例
   * @return int8_t 返回量化后的 int8 值
   */
  static int8_t QntFP32toAffine(float f32, int32_t zp, float scale);

  /**
   * @brief 对输入数组进行快速排序（按逆序索引排序）
   *
   * @param input 输入数组
   * @param left 左边界索引
   * @param right 右边界索引
   * @param indices 输出的排序后索引数组
   * @return int 返回排序状态码，0 表示成功
   */
  static int QuickSortIndiceInverse(std::vector<float> &input, int left,
                                    int right, std::vector<int> &indices);

  /**
   * @brief 非极大值抑制 (NMS)
   *
   * @param validCount 有效目标数
   * @param outputLocations 输出的目标位置 (x1, y1, x2, y2)
   * @param classIds 检测到的目标类别 ID
   * @param order 输入目标的排序顺序
   * @param filterId 要过滤的类别 ID
   * @param threshold 阈值，用于决定是否抑制目标框
   * @return int 返回处理状态码，0 表示成功
   */
  static int NMS(int validCount, std::vector<float> &outputLocations,
                 std::vector<int> classIds, std::vector<int> &order,
                 int filterId, float threshold);

  /**
   * @brief 计算两个矩形框的重叠度 (IoU)
   *
   * @param xmin0 第一个矩形框的左上角 x 坐标
   * @param ymin0 第一个矩形框的左上角 y 坐标
   * @param xmax0 第一个矩形框的右下角 x 坐标
   * @param ymax0 第一个矩形框的右下角 y 坐标
   * @param xmin1 第二个矩形框的左上角 x 坐标
   * @param ymin1 第二个矩形框的左上角 y 坐标
   * @param xmax1 第二个矩形框的右下角 x 坐标
   * @param ymax1 第二个矩形框的右下角 y 坐标
   * @return float 返回两个矩形框的 IoU 值
   */
  static float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                                float ymax0, float xmin1, float ymin1,
                                float xmax1, float ymax1);

  /**
   * @brief 将浮点值裁剪到指定整数范围
   *
   * @param val 输入的浮点值
   * @param min 最小值
   * @param max 最大值
   * @return int 返回裁剪后的整数值
   */
  inline static int Clamp(float val, int min, int max);

  // 锚框，在训练时已经确定
  static constexpr int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                                       {30, 61, 62, 45, 59, 119},
                                       {116, 90, 156, 198, 373, 326}};

  rknn_app_context_t rknn_app_ctx_;
  std::vector<std::string> labels_;
  // char *labels[OBJ_CLASS_NUM];
};
}  // namespace rknn_yolo_inference