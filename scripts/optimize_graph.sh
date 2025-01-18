#!/bin/bash
python scripts/optimize_graph.py --input_model ./models/yolov5s.onnx --output_model ./models/yolov5s-optimized.onnx \
  --nodes /model.24/m.0/Conv_output_0,/model.24/m.1/Conv_output_0,/model.24/m.2/Conv_output_0