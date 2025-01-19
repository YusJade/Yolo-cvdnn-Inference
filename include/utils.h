#include <fstream>
#include <string>
#include <vector>

inline std::vector<std::string> LoadLabels(std::string label_path) {
  std::vector<std::string> labels;
  std::ifstream file_stream(label_path);
  std::string label_name;

  labels.clear();
  while (std::getline(file_stream, label_name)) {
    labels.push_back(label_name);
  }

  return labels;
}