import loguru
import numpy as np
import onnx
from onnx import shape_inference
import argparse
import onnx_graphsurgeon as gs

# 命令行参数解析
parser = argparse.ArgumentParser(description="将节点的输出设置为模型输出(应用 Sigmoid 激活)")
parser.add_argument("--input_model", type=str, required=True,
                    help="输入 ONNX 模型路径")
parser.add_argument("--output_model", type=str, required=True,
                    help="输出修改后的 ONNX 模型路径")
parser.add_argument("--nodes", type=str, required=True,
                    help="节点输出列表，用逗号分隔，例如: 1,3")
args = parser.parse_args()

# 加载模型
model_path = args.input_model
output_path = args.output_model

nodes = args.nodes.split(',')
loguru.logger.info(nodes)
model = onnx.load(model_path)
graph = gs.import_onnx(model)

# 解析输出名称
output_names = nodes

# 添加 Sigmoid 节点
new_outputs = []
for idx in range(len(output_names)):
    output_name = output_names[idx]
    # 获取对应的张量
    tensor = graph.tensors()[output_name]

    # 创建 Sigmoid 节点
    sigmoid_tensor = gs.Variable(f"output{idx}", dtype=np.float32)
    sigmoid_node = gs.Node(
        op="Sigmoid",
        inputs=[tensor],
        outputs=[sigmoid_tensor]
    )

    # 将节点加入图中
    graph.nodes.append(sigmoid_node)
    new_outputs.append(sigmoid_tensor)

# 更新图的输出
graph.outputs = new_outputs

# 清理并拓扑排序
graph.cleanup().toposort()

# 导出修改后的模型
onnx_model = gs.export_onnx(graph)

# 推断新模型的形状信息
onnx_model = shape_inference.infer_shapes(onnx_model)

# 保存模型
onnx.save(onnx_model, output_path)
print(f"已保存修改后的模型到 {output_path}")
