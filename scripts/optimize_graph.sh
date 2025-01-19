#!/bin/bash
python scripts/optimize_graph.py --input_model models/v5lite-s-exported_by_v5.onnx --output_model models/v5lite-s-optimized-exported_by_v5.onnx \
  --nodes /model.21/m.0/Conv_output_0,/model.21/m.1/Conv_output_0,/model.21/m.2/Conv_output_0