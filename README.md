# MPI Dijkstra Parallel Optimization with CI/CD

Dự án này song song hóa thuật toán Dijkstra bằng thư viện MPI/OpenMPI, áp dụng các kỹ thuật tối ưu hóa mức thấp bao gồm: nén bit dữ liệu truyền thông, hàng đợi ưu tiên kiểu nguyên 64-bit và cấu trúc đồ thị CSR phẳng liên tục.

## 1. Yêu cầu hệ thống (Yêu cầu cục bộ)
Để chạy thử nghiệm ứng dụng này trên máy tính cá nhân (Ubuntu / WSL), bạn cần cài đặt:
- Trình biên dịch C++ hỗ trợ C++11 trở lên.
- Thư viện OpenMPI hoặc MPICH.

Lệnh cài đặt trên Ubuntu/WSL:
```bash
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev