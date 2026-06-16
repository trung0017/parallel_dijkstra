#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

const long long INF = numeric_limits<long long>::max();
const int V = 10000000;      // 10 triệu đỉnh
const int E = 100000000;     // 100 triệu cạnh
const int SOURCE_NODE = 0;   // Đỉnh bắt đầu

// Cấu trúc đồ thị: Danh sách kề (Adjacency List)
// adj[u] chứa các cặp {v, weight}
vector<vector<pair<int, long long>>> adj;

// Hàm sinh đồ thị ngẫu nhiên sử dụng cùng seed trên mọi tiến trình
void generateGraph() {
    adj.resize(V);
    mt19937 gen(42); // Seed 42 cố định
    uniform_int_distribution<int> dist_v(0, V - 1);
    uniform_int_distribution<long long> dist_w(1, 100);

    for (int i = 0; i < E; ++i) {
        int u = dist_v(gen);
        int v = dist_v(gen);
        long long w = dist_w(gen);
        if (u != v) {
            adj[u].push_back({v, w});
            // Đồ thị có hướng (nếu vô hướng thì thêm adj[v].push_back({u, w}))
        }
    }
}

// Hàm Dijkstra tuần tự (Sử dụng Priority Queue tối ưu O(E log V))
double sequential_dijkstra() {
    vector<long long> dist(V, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;

    auto start_time = chrono::high_resolution_clock::now();

    dist[SOURCE_NODE] = 0;
    pq.push({0, SOURCE_NODE});

    while (!pq.empty()) {
        auto top = pq.top();
        pq.pop();
        long long d = top.first;
        int u = top.second;

        if (d > dist[u]) continue;

        for (auto& edge : adj[u]) {
            int v = edge.first;
            long long weight = edge.second;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;
    return diff.count();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Khởi tạo đồ thị đồng nhất trên tất cả các tiến trình
    if (rank == 0) cout << "[Master] Dang sinh do thi ngau nhien..." << endl;
    generateGraph();
    MPI_Barrier(MPI_COMM_WORLD);

    double seq_time = 0.0;
    
    // 2. Chạy bản tuần tự trên Master để làm cơ sở đo lường
    if (rank == 0) {
        cout << "[Master] Bat dau chay Dijkstra Tuan tu (Sequential)..." << endl;
        seq_time = sequential_dijkstra();
        cout << "[Master] Thoi gian Tuan tu: " << fixed << setprecision(4) << seq_time << " giay\n" << endl;
    }

    // 3. Chuẩn bị phân chia công việc cho MPI
    // Chia đều các đỉnh cho các tiến trình (1D Block Mapping)
    int local_n = V / size;
    int remainder = V % size;
    int start_v = rank * local_n + min(rank, remainder);
    int end_v = start_v + local_n + (rank < remainder ? 1 : 0);
    int local_count = end_v - start_v;

    // Chuẩn bị mảng recvcounts và displs cho Scatterv và Gatherv
    vector<int> counts(size), displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            int count_i = V / size + (i < remainder ? 1 : 0);
            counts[i] = count_i;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }
    }

    vector<long long> global_init_dist;
    vector<long long> local_dist(local_count, INF);
    vector<bool> local_visited(local_count, false);

    if (rank == 0) {
        global_init_dist.assign(V, INF);
        global_init_dist[SOURCE_NODE] = 0;
        cout << "[Master] Bat dau chay Dijkstra Song song (MPI)..." << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto par_start_time = chrono::high_resolution_clock::now();

    // SỬ DỤNG MPI_Scatterv ĐỂ ĐIỀU PHỐI DỮ LIỆU BAN ĐẦU
    MPI_Scatterv(rank == 0 ? global_init_dist.data() : nullptr, 
                 rank == 0 ? counts.data() : nullptr, 
                 rank == 0 ? displs.data() : nullptr, 
                 MPI_LONG_LONG, 
                 local_dist.data(), local_count, MPI_LONG_LONG, 
                 0, MPI_COMM_WORLD);

    // Mảng cấu hình cho việc Gatherv thu thập local_min
    vector<int> min_counts(size, 2); // Mỗi process gửi 2 phần tử (dist, node_id)
    vector<int> min_displs(size);
    if (rank == 0) {
        for (int i = 0; i < size; ++i) min_displs[i] = i * 2;
    }
    vector<long long> gathered_mins(size * 2);

    // Vòng lặp duyệt Dijkstra song song (Dạng mảng - Array based)
    for (int step = 0; step < V; ++step) {
        
        // In trạng thái định kỳ để tránh chương trình có vẻ như bị "treo"
        if (rank == 0 && step % 1000 == 0) {
            cout << "[Trang thai] Da duyet " << step << " / " << V << " dinh..." << endl;
            cout.flush();
        }

        // Bước 3.1: Tìm đỉnh có khoảng cách nhỏ nhất ở local
        long long local_min_dist = INF;
        long long local_min_u = -1;
        for (int i = 0; i < local_count; ++i) {
            if (!local_visited[i] && local_dist[i] < local_min_dist) {
                local_min_dist = local_dist[i];
                local_min_u = start_v + i;
            }
        }

        // Đóng gói dữ liệu cơ bản thành mảng để truyền đi (tránh lỗi struct/Allreduce)
        long long send_min_data[2] = {local_min_dist, local_min_u};

        // SỬ DỤNG MPI_Gatherv ĐỂ MASTER THU THẬP KẾT QUẢ TỪ WORKER
        MPI_Gatherv(send_min_data, 2, MPI_LONG_LONG,
                    rank == 0 ? gathered_mins.data() : nullptr,
                    rank == 0 ? min_counts.data() : nullptr,
                    rank == 0 ? min_displs.data() : nullptr,
                    MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        long long global_min[2] = {INF, -1};

        // Bước 3.2: Master tìm đỉnh nhỏ nhất toàn cục
        if (rank == 0) {
            for (int p = 0; p < size; ++p) {
                if (gathered_mins[p * 2] < global_min[0]) {
                    global_min[0] = gathered_mins[p * 2];       // Khoảng cách
                    global_min[1] = gathered_mins[p * 2 + 1];   // Node ID
                }
            }
        }

        // Bước 3.3: Broadcast đỉnh nhỏ nhất cho tất cả Workers
        MPI_Bcast(global_min, 2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        if (global_min[0] == INF) break; // Toàn bộ đồ thị liên thông đã duyệt xong

        int u = static_cast<int>(global_min[1]);
        
        // Đánh dấu đã thăm nếu đỉnh thuộc tiến trình này
        if (u >= start_v && u < end_v) {
            local_visited[u - start_v] = true;
        }

        // Bước 3.4: Cập nhật khoảng cách. 
        // Vì mọi tiến trình đều có chung biến `adj` (do sinh ngẫu nhiên cùng seed),
        // mọi tiến trình đều có thể truy cập adj[u] ngay lập tức.
        for (auto& edge : adj[u]) {
            int v = edge.first;
            long long weight = edge.second;
            
            // Chỉ cập nhật nếu v thuộc vùng quản lý của tiến trình hiện tại
            if (v >= start_v && v < end_v) {
                int local_v_idx = v - start_v;
                if (!local_visited[local_v_idx] && global_min[0] + weight < local_dist[local_v_idx]) {
                    local_dist[local_v_idx] = global_min[0] + weight;
                }
            }
        }
    }

    // SỬ DỤNG MPI_Gatherv ĐỂ MASTER TỔNG HỢP MẢNG KHOẢNG CÁCH CUỐI CÙNG
    vector<long long> final_dist;
    if (rank == 0) final_dist.resize(V);

    MPI_Gatherv(local_dist.data(), local_count, MPI_LONG_LONG,
                rank == 0 ? final_dist.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto par_end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> par_diff = par_end_time - par_start_time;
    double par_time = par_diff.count();

    // 4. In kết quả Speedup và Efficiency trên Master
    if (rank == 0) {
        double speedup = seq_time / par_time;
        double efficiency = speedup / size;

        cout << "\n--- KET QUA DANH GIA HIEU NANG ---" << endl;
        cout << "So tien trinh (p): " << size << endl;
        cout << "Thoi gian tuan tu (T1): " << fixed << setprecision(4) << seq_time << " giay" << endl;
        cout << "Thoi gian song song (Tp): " << fixed << setprecision(4) << par_time << " giay" << endl;
        cout << "Speedup S(p) = " << fixed << setprecision(4) << speedup << endl;
        cout << "Efficiency E(p) = " << fixed << setprecision(4) << efficiency << endl;
    }

    MPI_Finalize();
    return 0;
}