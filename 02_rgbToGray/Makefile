# 编译器
NVCC = nvcc

# 编译参数（OpenCV 路径由 pkg-config 自动获取）
CFLAGS = -std=c++17 -arch=sm_75 $(shell pkg-config --cflags opencv4)
LIBS   = $(shell pkg-config --libs opencv4)

# 目标可执行文件名
TARGET = main

# 源文件
SRC = main.cu

# 默认执行：编译并运行
all: $(TARGET)
	./$(TARGET)

# 编译规则
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) $(LIBS) -o $(TARGET)

# 清理
clean:
	rm -f $(TARGET)

.PHONY: all clean