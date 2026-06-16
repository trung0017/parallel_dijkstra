CXX = mpic++
CXXFLAGS = -O3 -Wall -Wextra

# Tham số tối ưu chạy thử nghiệm nhanh cho CI (50k đỉnh, 500k cạnh)
CI_FLAGS = -DV_MAX=50000 -DE_MAX=500000

TARGET = parallel_dijkstra_opt
SRC = src/parallel_dijkstra_opt.cpp

# Chế độ mặc định (Mục tiêu chính cho báo cáo quy mô lớn - 10 triệu đỉnh)
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Biên dịch tối ưu riêng cho CI/CD test nhanh để không bị treo máy ảo GitHub
ci-build: $(SRC)
	$(CXX) $(CXXFLAGS) $(CI_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)