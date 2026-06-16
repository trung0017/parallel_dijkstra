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

// Định nghĩa kích thước đồ thị thông qua Macro
#ifndef V_MAX
#define V_MAX 10000000       // Mặc định: 10 triệu đỉnh cho báo cáo lớn
#endif

#ifndef E_MAX
#define E_MAX 100000000      // Mặc định: 100 triệu cạnh cho báo cáo lớn
#endif

// Nén khoảng cách (32-bit) và đỉnh (32-bit) vào một số nguyên 64-bit để tối ưu hóa truyền tin
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

        vector<int> degree(V, 0);
        for (int i = 0; i < V - 1; ++i) {
            degree[i]++;
        }
        int remaining = E - (V - 1);
        vector<pair<int, int>> temp_edges;
        temp_edges.reserve(remaining);
        for (int i = 0; i < remaining; ++i) {
            int u = rand.next_range(0, V - 1);
            int v = rand.next_range(0, V - 1);
            degree[u]++;
            temp_edges.push_back({u, v});
        }

        head[0] = 0;
        for (int i = 0; i < V; ++i) {
            head[i+1] = head[i] + degree[i];
        }

        vector<int> current_pos = head;

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
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> pq;

    dist[source] = 0;
    pq.push(pack(0, source));

    const int* graph_head = graph.head.data();
    const int* graph_to = graph.to.data();
    const int* graph_weight = graph.weight.data();

    while (!pq.empty()) {
        uint64_t top = pq.top();
        pq.pop();

        uint32_t d, u;
        unpack(top, d, u);

        if (d > dist[u]) continue;

        int edge_start = graph_head[u];
        int edge_end = graph_head[u + 1];
        for (int e = edge_start; e < edge_end; ++e) {
            int v = graph_to[e];
            int w = graph_weight[e];
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push(pack(dist[v], v));
            }
        }
    }
    return dist;
}

// --- PHIÊN BẢN SONG SONG DIJKSTRA THEO PHƯƠNG PHÁP CHUẨN TRONG TÀI LIỆU ---
vector<uint32_t> dijkstra_parallel(const CSRGraph& graph, int source, int rank, int size) {
    int V = graph.V;

    // Phân hoạch dữ liệu (Tài liệu Trang 23 - Section 2.3)
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

    // Khởi tạo mảng khoảng cách cục bộ (Tài liệu Trang 23 - Section 2.3)
    vector<uint32_t> global_dist;
    if (rank == 0) {
        global_dist.assign(V, INF);
        global_dist[source] = 0;
    }

    vector<uint32_t> local_dist(local_N);
    MPI_Scatterv(global_dist.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                 local_dist.data(), local_N, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Mảng đánh dấu marker đã duyệt (Tài liệu Trang 51)
    vector<bool> local_visited(local_N, false);
    
    // Tối ưu hóa tìm cực tiểu bằng Priority Queue (Đề xuất tối ưu hóa tại Tài liệu Trang 42 - Chương 5)
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> local_pq;

    if (source >= start_v && source < end_v) {
        local_pq.push(pack(0, source));
    }

    const int* p_graph_head = graph.head.data();
    const int* p_graph_to = graph.to.data();
    const int* p_graph_weight = graph.weight.data();

    int print_interval = (V >= 10000) ? (V / 10) : 1000;

    // Vòng lặp chính chạy đúng V bước tuần tự chặt chẽ (Tài liệu Trang 51 - Main loop)
    for (int iter = 0; iter < V; ++iter) {
        uint64_t local_min = pack(INF, 0xFFFFFFFFULL);

        // Bước 1: Tìm đỉnh có khoảng cách nhỏ nhất trong phân vùng cục bộ (Tài liệu Trang 51 - Step 1)
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

        // Bước 2: Tìm kiếm cực tiểu toàn cục sử dụng Allreduce (Tài liệu Trang 52 - Step 2)
        // (Sử dụng uint64_t nén để tránh lỗi biên dịch cấu trúc MPI_MINLOC phức tạp)
        uint64_t global_min;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);

        uint32_t global_min_dist, global_min_vertex;
        unpack(global_min, global_min_dist, global_min_vertex);

        if (global_min_dist == INF) {
            break; 
        }

        // Đánh dấu đỉnh tối ưu toàn cục đã duyệt (Tài liệu Trang 52 - Step 2)
        if (global_min_vertex >= start_v && global_min_vertex < end_v) {
            local_visited[global_min_vertex - start_v] = true;
            if (!local_pq.empty()) {
                local_pq.pop();
            }
        }

        // Bước 3: Cập nhật khoảng cách nhãn tạm thời cho phân vùng cục bộ (Tài liệu Trang 52 - Step 3)
        int edge_start = p_graph_head[global_min_vertex];
        int edge_end = p_graph_head[global_min_vertex + 1];
        for (int e = edge_start; e < edge_end; ++e) {
            int v = p_graph_to[e];
            int w = p_graph_weight[e];
            if (v >= start_v && v < end_v) {
                int local_v = v - start_v;
                if (!local_visited[local_v]) {
                    uint32_t new_dist = global_min_dist + w;
                    if (new_dist < local_dist[local_v]) {
                        local_dist[local_v] = new_dist;
                        local_pq.push(pack(new_dist, v));
                    }
                }
            }
        }

        if (rank == 0 && iter % print_interval == 0) {
            cout << "[Master] Vong lap Dijkstra " << iter << " / " << V 
                 << " | Dinh toi uu hien tai: " << global_min_vertex 
                 << " | Chi phi: " << global_min_dist << "\n";
            cout.flush();
        }
    }

    // Thu thập mảng khoảng cách cuối cùng về Master bằng MPI_Gatherv (Tài liệu Trang 55)
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
        cout << " KHAO SAT DIJKSTRA SONG SONG MPI THEO PHUONG PHAP TAI LIEU\n";
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