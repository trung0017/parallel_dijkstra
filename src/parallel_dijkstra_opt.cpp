#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <climits>
#include <iomanip>
#include <mpi.h>

using namespace std;

const uint32_t INF = 2000000000; 

#ifndef V_MAX
#define V_MAX 10000000       // Mặc định: 10 triệu đỉnh cho báo cáo lớn
#endif

#ifndef E_MAX
#define E_MAX 100000000      // Mặc định: 100 triệu cạnh cho báo cáo lớn
#endif

inline uint64_t pack(uint32_t dist, uint32_t vertex) {
    return ((uint64_t)dist << 32) | vertex;
}

inline void unpack(uint64_t packed, uint32_t& dist, uint32_t& vertex) {
    dist = packed >> 32;
    vertex = packed & 0xFFFFFFFFULL;
}

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

// --- SEQUENTIAL DIJKSTRA ---
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

// --- PARALLEL SSSP VOI GIAI THUAT HYBRID BSP-SPFA & TARGET-PARTITIONED CSR ---
vector<uint32_t> dijkstra_parallel(const CSRGraph& graph, int source, int rank, int size) {
    int V = graph.V;

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

    // Phân hoạch đồ thị theo phân vùng đích (Target-Partitioned CSR)
    vector<int> local_head(V + 1, 0);
    vector<int> local_to;
    vector<int> local_weight;
    
    vector<int> local_degree(V, 0);
    for (int u = 0; u < V; ++u) {
        for (int e = graph.head[u]; e < graph.head[u + 1]; ++e) {
            int v = graph.to[e];
            if (v >= start_v && v < end_v) {
                local_degree[u]++;
            }
        }
    }

    local_head[0] = 0;
    for (int i = 0; i < V; ++i) {
        local_head[i + 1] = local_head[i] + local_degree[i];
    }

    local_to.resize(local_head[V]);
    local_weight.resize(local_head[V]);

    vector<int> local_pos = local_head;
    for (int u = 0; u < V; ++u) {
        for (int e = graph.head[u]; e < graph.head[u + 1]; ++e) {
            int v = graph.to[e];
            int w = graph.weight[e];
            if (v >= start_v && v < end_v) {
                int pos = local_pos[u]++;
                local_to[pos] = v;
                local_weight[pos] = w;
            }
        }
    }

    // Mỗi tiến trình duy trì mảng khoảng cách toàn cục dist (chỉ tốn ~40 MB RAM cho 10 triệu đỉnh)
    vector<uint32_t> dist(V, INF);
    dist[source] = 0;

    // Hàng đợi cục bộ và các mảng đánh dấu trạng thái hàng đợi để loại bỏ trùng lặp
    queue<int> local_queue;
    vector<bool> in_queue(local_N, false);
    vector<bool> is_in_update_list(local_N, false);

    if (source >= start_v && source < end_v) {
        local_queue.push(source);
        in_queue[source - start_v] = true;
    }

    const int* p_local_head = local_head.data();
    const int* p_local_to = local_to.data();
    const int* p_local_weight = local_weight.data();

    int sync_step = 0;

    // --- VÒNG LẶP CHÍNH BULK SYNCHRONOUS SPFA ---
    while (true) {
        vector<int> updated_local_vertices;

        // PHA 1: Tính toán cục bộ không chặn (Local SPFA) - HOÀN TOÀN KHÔNG TRUYỀN TIN
        while (!local_queue.empty()) {
            int u = local_queue.front();
            local_queue.pop();
            int local_u = u - start_v;
            in_queue[local_u] = false;

            int edge_start = p_local_head[u];
            int edge_end = p_local_head[u + 1];
            for (int e = edge_start; e < edge_end; ++e) {
                int v = p_local_to[e];
                int w = p_local_weight[e];
                int local_v = v - start_v;

                if (dist[u] + w < dist[local_v]) {
                    dist[local_v] = dist[u] + w;
                    
                    if (!in_queue[local_v]) {
                        local_queue.push(v);
                        in_queue[local_v] = true;
                    }
                    if (!is_in_update_list[local_v]) {
                        updated_local_vertices.push_back(v);
                        is_in_update_list[local_v] = true;
                    }
                }
            }
        }

        // Chuẩn bị dữ liệu cập nhật để đồng bộ
        vector<uint64_t> local_updates;
        for (int v : updated_local_vertices) {
            local_updates.push_back(pack(dist[v], v));
            is_in_update_list[v - start_v] = false; // Reset trạng thái
        }

        int local_size = local_updates.size();
        vector<int> recvcounts(size);
        
        // PHA 2: Đồng bộ hóa định kỳ (Lazy Bulk Synchronization)
        MPI_Allgather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> displs(size);
        int total_updates = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total_updates;
            total_updates += recvcounts[i];
        }

        // Thuật toán hội tụ khi không còn tiến trình nào phát sinh cập nhật mới
        if (total_updates == 0) {
            break; 
        }

        vector<uint64_t> global_updates(total_updates);
        MPI_Allgatherv(local_updates.data(), local_size, MPI_UINT64_T,
                       global_updates.data(), recvcounts.data(), displs.data(), MPI_UINT64_T,
                       MPI_COMM_WORLD);

        // Áp dụng các cập nhật toàn cục nhận được và kích hoạt hàng đợi cho pha tiếp theo
        for (int i = 0; i < total_updates; ++i) {
            uint32_t d_u, u;
            unpack(global_updates[i], d_u, u);

            dist[u] = d_u;

            int edge_start = p_local_head[u];
            int edge_end = p_local_head[u + 1];
            for (int e = edge_start; e < edge_end; ++e) {
                int v = p_local_to[e];
                int w = p_local_weight[e];
                int local_v = v - start_v;

                if (dist[u] + w < dist[local_v]) {
                    dist[local_v] = dist[u] + w;
                    if (!in_queue[local_v]) {
                        local_queue.push(v);
                        in_queue[local_v] = true;
                    }
                }
            }
        }

        if (rank == 0 && sync_step % 10 == 0) {
            cout << "[Master] Buoc dong bo " << sync_step << " | So luong cap nhat toan cuc: " << total_updates << "\n";
            cout.flush();
        }
        sync_step++;
    }

    // Thu thập kết quả mảng khoảng cách cuối cùng về Master bằng MPI_Gatherv
    vector<uint32_t> final_global_dist;
    if (rank == 0) {
        final_global_dist.resize(V);
    }

    vector<uint32_t> local_dist(local_N);
    for (int i = 0; i < local_N; ++i) {
        local_dist[i] = dist[start_v + i];
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
        cout << " KHAO SAT SSSP SONG SONG MPI THEO PHUONG PHAP HYBRID BSP-SPFA\n";
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
        cout << "[Master] Dang chay phien ban Song song (Parallel SSSP)...\n";
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