#!/bin/bash
python scripts/optimize_graph.py --input_model ./best07.onnx --output_model ./best07-optimized.onnx \
  --nodes /model.21/m.0/Conv_output_0,/model.21/m.1/Conv_output_0,/model.21/m.2/Conv_output_0