#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <climits>
#include <iomanip>
#include <mpi.h>

using namespace std;

// Khoảng cách vô cùng tối ưu cho kiểu uint32_t (tránh tràn số khi cộng trọng số)
const uint32_t INF = 2000000000; 

// Định nghĩa kích thước đồ thị thông qua Macro để linh hoạt khi chạy CI/CD và chạy báo cáo thực tế
#ifndef V_MAX
#define V_MAX 10000000       // Mặc định: 10 triệu đỉnh cho báo cáo lớn
#endif

#ifndef E_MAX
#define E_MAX 100000000      // Mặc định: 100 triệu cạnh cho báo cáo lớn
#endif

// Kỹ thuật nén khoảng cách (32-bit) và đỉnh (32-bit) vào một số nguyên 64-bit để giảm 50% số cuộc gọi MPI
inline uint64_t pack(uint32_t dist, uint32_t vertex) {
    return ((uint64_t)dist << 32) | vertex;
}

// Giải nén số nguyên 64-bit thành khoảng cách và đỉnh tương ứng
inline void unpack(uint64_t packed, uint32_t& dist, uint32_t& vertex) {
    dist = packed >> 32;
    vertex = packed & 0xFFFFFFFFULL;
}

// Bộ sinh số ngẫu nhiên Xorshift64 đồng nhất nhanh trên các tiến trình
struct FastRand {
    uint64_t state;
    FastRand(uint64_t seed) : state(seed) {}
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (state * 0x2545F4914F6CDD1DULL) >> 32;
    }
    int next_range(int l, int r) {
        return l + (next() % (r - l + 1));
    }
};

// Cấu trúc đồ thị CSR phẳng tối ưu bộ nhớ đệm (Cache-friendly)
struct CSRGraph {
    int V;
    int E;
    vector<int> head;
    vector<int> to;
    vector<int> weight;

    void generate(int num_vertices, int num_edges, uint64_t seed) {
        V = num_vertices;
        E = num_edges;
        head.assign(V + 1, 0);
        to.resize(E);
        weight.resize(E);

        FastRand rand(seed);

        // Đếm bậc của các đỉnh để xây dựng mảng head
        vector<int> degree(V, 0);
        // 1. Đảm bảo tính liên thông bằng đồ thị đường thẳng
        for (int i = 0; i < V - 1; ++i) {
            degree[i]++;
        }
        // 2. Sinh ngẫu nhiên các cạnh còn lại
        int remaining = E - (V - 1);
        vector<pair<int, int>> temp_edges;
        temp_edges.reserve(remaining);
        for (int i = 0; i < remaining; ++i) {
            int u = rand.next_range(0, V - 1);
            int v = rand.next_range(0, V - 1);
            degree[u]++;
            temp_edges.push_back({u, v});
        }

        // Tạo mảng head bằng tổng tích lũy
        head[0] = 0;
        for (int i = 0; i < V; ++i) {
            head[i+1] = head[i] + degree[i];
        }

        // Mảng tạm lưu vị trí chèn hiện tại của từng đỉnh
        vector<int> current_pos = head;

        // Điền dữ liệu cạnh vào các mảng phẳng liên tục
        for (int i = 0; i < V - 1; ++i) {
            int u = i;
            int v = i + 1;
            int w = rand.next_range(1, 100);
            int pos = current_pos[u]++;
            to[pos] = v;
            weight[pos] = w;
        }
        for (int i = 0; i < remaining; ++i) {
            int u = temp_edges[i].first;
            int v = temp_edges[i].second;
            int w = rand.next_range(1, 100);
            int pos = current_pos[u]++;
            to[pos] = v;
            weight[pos] = w;
        }
    }
};

// --- PHIÊN BẢN TUẦN TỰ (SEQUENTIAL DIJKSTRA) ---
vector<uint32_t> dijkstra_sequential(const CSRGraph& graph, int source) {
    int V = graph.V;
    vector<uint32_t> dist(V, INF);
    // Sử dụng hàng đợi ưu tiên lưu số nguyên 64-bit nén để tối ưu tốc độ so sánh
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> pq;

    dist[source] = 0;
    pq.push(pack(0, source));

    while (!pq.empty()) {
        uint64_t top = pq.top();
        pq.pop();

        uint32_t d, u;
        unpack(top, d, u);

        if (d > dist[u]) continue;

        // Duyệt lân cận liên tục trong bộ nhớ
        for (int e = graph.head[u]; e < graph.head[u + 1]; ++e) {
            int v = graph.to[e];
            int w = graph.weight[e];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push(pack(dist[v], v));
            }
        }
    }
    return dist;
}

// --- PHIÊN BẢN SONG SONG MPI (PARALLEL DIJKSTRA) ---
vector<uint32_t> dijkstra_parallel(const CSRGraph& graph, int source, int rank, int size) {
    int V = graph.V;

    // Phân chia cân bằng tải các đỉnh cho các tiến trình
    vector<int> sendcounts(size);
    vector<int> displs(size);
    int chunk = V / size;
    int rem = V % size;
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = chunk + (i < rem ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_N = sendcounts[rank];
    int start_v = displs[rank];
    int end_v = start_v + local_N;

    // Khởi tạo mảng khoảng cách toàn cục trên Master
    vector<uint32_t> global_dist;
    if (rank == 0) {
        global_dist.assign(V, INF);
        global_dist[source] = 0;
    }

    // 1. Sử dụng MPI_Scatterv để phân phối trạng thái ban đầu
    vector<uint32_t> local_dist(local_N);
    MPI_Scatterv(global_dist.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                 local_dist.data(), local_N, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    vector<bool> local_visited(local_N, false);
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> local_pq;

    if (source >= start_v && source < end_v) {
        local_pq.push(pack(0, source));
    }

    // Khoảng thời gian in tiến trình dựa trên kích thước đồ thị thực tế
    int print_interval = (V >= 10000) ? (V / 10) : 1000;

    for (int iter = 0; iter < V; ++iter) {
        uint64_t local_min = pack(INF, 0xFFFFFFFFULL);

        // Tìm kiếm đỉnh cục bộ nhỏ nhất bằng Heap tối ưu
        while (!local_pq.empty()) {
            uint64_t top = local_pq.top();
            uint32_t d, u;
            unpack(top, d, u);

            int local_u = u - start_v;
            if (local_visited[local_u]) {
                local_pq.pop();
                continue;
            }
            if (d > local_dist[local_u]) {
                local_pq.pop();
                continue;
            }
            local_min = top;
            break;
        }

        // Khởi tạo vùng đệm thu gom trên Master
        vector<uint64_t> gathered;
        if (rank == 0) {
            gathered.resize(size);
        }

        // GOM CHUNG: gom đồng thời cả khoảng cách và ID đỉnh thông qua biến số nguyên 64-bit nén
        MPI_Gather(&local_min, 1, MPI_UINT64_T, gathered.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        uint64_t global_min = pack(INF, 0xFFFFFFFFULL);
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                if (gathered[i] < global_min) {
                    global_min = gathered[i];
                }
            }
        }

        // PHÁT SÓNG CHUNG: phát đồng thời trạng thái nhỏ nhất toàn cục
        MPI_Bcast(&global_min, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        uint32_t global_min_dist, global_min_vertex;
        unpack(global_min, global_min_dist, global_min_vertex);

        if (global_min_dist == INF) {
            break;
        }

        // Đánh dấu đã duyệt nếu thuộc tiến trình quản lý
        if (global_min_vertex >= start_v && global_min_vertex < end_v) {
            local_visited[global_min_vertex - start_v] = true;
        }

        // Cập nhật khoảng cách cục bộ cho các lân cận của đỉnh vừa duyệt
        for (int e = graph.head[global_min_vertex]; e < graph.head[global_min_vertex + 1]; ++e) {
            int v = graph.to[e];
            int w = graph.weight[e];
            if (v >= start_v && v < end_v) {
                int local_v = v - start_v;
                if (!local_visited[local_v]) {
                    if (global_min_dist + w < local_dist[local_v]) {
                        local_dist[local_v] = global_min_dist + w;
                        local_pq.push(pack(local_dist[local_v], v));
                    }
                }
            }
        }

        // In trạng thái định kỳ giãn cách tránh nghẽn luồng xuất màn hình của OS
        if (rank == 0 && iter % print_interval == 0) {
            cout << "[Master] Tien do: " << iter << " / " << V 
                 << " | Dinh hien tai: " << global_min_vertex 
                 << " | Chi phi: " << global_min_dist << "\n";
            cout.flush();
        }
    }

    // 2. Sử dụng MPI_Gatherv thu thập kết quả cuối cùng từ các Workers về Master
    vector<uint32_t> final_global_dist;
    if (rank == 0) {
        final_global_dist.resize(V);
    }

    MPI_Gatherv(local_dist.data(), local_N, MPI_UINT32_T,
                final_global_dist.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                0, MPI_COMM_WORLD);

    return final_global_dist;
}

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int source = 0; 

    if (rank == 0) {
        cout << "========================================================\n";
        cout << " KHAO SAT DIJKSTRA SONG SONG MPI (PHAS TOI UU CAP CAO)\n";
        cout << " Quy mo do thi: " << V_MAX << " dinh | " << E_MAX << " canh\n";
        cout << " So luong tien trinh (MPI Processes): " << size << "\n";
        cout << "========================================================\n";
        cout << "[Master] Dang khoi tao do thi CSR lien tuc trong RAM...\n";
        cout.flush();
    }

    CSRGraph graph;
    auto t_gen_start = chrono::high_resolution_clock::now();
    graph.generate(V_MAX, E_MAX, 42);
    auto t_gen_end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        double gen_time = chrono::duration<double>(t_gen_end - t_gen_start).count();
        cout << "[Master] Tao do thi thanh cong trong: " << fixed << setprecision(4) << gen_time << " giay.\n";
        cout << "[Master] Dang chay phien ban Tuan tu (Sequential Dijkstra)...\n";
        cout.flush();
    }

    vector<uint32_t> seq_dist;
    double t_seq = 0.0;
    if (rank == 0) {
        auto t_seq_start = chrono::high_resolution_clock::now();
        seq_dist = dijkstra_sequential(graph, source);
        auto t_seq_end = chrono::high_resolution_clock::now();
        t_seq = chrono::duration<double>(t_seq_end - t_seq_start).count();
        cout << "[Master] Hoan thanh Tuan tu trong: " << t_seq << " giay.\n";
        cout << "[Master] Dang chay phien ban Song song (Parallel Dijkstra)...\n";
        cout.flush();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_par_start = chrono::high_resolution_clock::now();
    
    vector<uint32_t> par_dist = dijkstra_parallel(graph, source, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_par_end = chrono::high_resolution_clock::now();
    double t_par = chrono::duration<double>(t_par_end - t_par_start).count();

    if (rank == 0) {
        cout << "[Master] Hoan thanh Song song trong: " << t_par << " giay.\n";
        
        bool correct = true;
        for (int i = 0; i < graph.V; ++i) {
            if (seq_dist[i] != par_dist[i]) {
                correct = false;
                break;
            }
        }

        cout << "\n================ KET QUA DANH GIA HE SO =================\n";
        if (correct) {
            cout << " KET QUA KIEU CHUNG: CHINH XAC\n";
        } else {
            cout << " KET QUA KIEU CHUNG: THAT BAI (Sai lech ket qua!)\n";
        }

        double speedup = t_seq / t_par;
        double efficiency = speedup / size;

        cout << " Thoi gian Tuan tu T(1) : " << t_seq << " giay\n";
        cout << " Thoi gian Song song T(p): " << t_par << " giay\n";
        cout << " Gia toc Speedup S(" << size << ")   : " << speedup << "\n";
        cout << " Hieu suat Efficiency E(" << size << "): " << efficiency * 100.0 << " %\n";
        cout << "========================================================\n";
        cout.flush();
    }

    MPI_Finalize();
    return 0;
}