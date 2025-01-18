#pragma once

#include <cstdint>

inline int32_t __clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

inline int8_t QntFP32toAffine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

inline uint8_t QntFP32toAffineUInt8(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
  return res;
}

inline float DeqntAffinetoFP32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

inline float DeqntAffineUInt8toFP32(uint8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}