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

// Hằng số độ rộng phân lô delta (Có thể điều chỉnh để tối ưu hóa hiệu năng)
const uint32_t DELTA = 50000; 

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

// --- PARALLEL SSSP VOI GIAI THUAT DELTA-STEPPING CHUAN HOA ---
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

    // Khởi tạo mảng khoảng cách toàn cục trên Master
    vector<uint32_t> global_dist;
    if (rank == 0) {
        global_dist.assign(V, INF);
        global_dist[source] = 0;
    }

    vector<uint32_t> local_dist(local_N);
    MPI_Scatterv(global_dist.data(), sendcounts.data(), displs.data(), MPI_UINT32_T,
                 local_dist.data(), local_N, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    vector<bool> local_visited(local_N, false);
    priority_queue<uint64_t, vector<uint64_t>, greater<uint64_t>> local_pq;

    if (source >= start_v && source < end_v) {
        local_pq.push(pack(0, source));
    }

    const int* p_local_head = local_head.data();
    const int* p_local_to = local_to.data();
    const int* p_local_weight = local_weight.data();

    int bucket_step = 0;
    
    // Mảng đánh dấu hoạt động để loại bỏ trùng lặp phần tử
    vector<bool> in_active_list(local_N, false);

    // --- VÒNG LẶP CHÍNH CỦA ĐỒNG BỘ PHÂN LÔ (DELTA-STEPPING) ---
    while (true) {
        uint32_t local_min_dist = INF;
        
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
            local_min_dist = d;
            break;
        }

        uint32_t global_min_dist;
        MPI_Allreduce(&local_min_dist, &global_min_dist, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);

        if (global_min_dist == INF) {
            break; 
        }

        uint32_t threshold = global_min_dist + DELTA;
        
        // Gom các đỉnh cục bộ thỏa mãn d <= threshold vào cụm hoạt động ban đầu
        vector<uint64_t> local_active;
        for (int i = 0; i < local_N; ++i) {
            if (!local_visited[i] && local_dist[i] <= threshold) {
                local_active.push_back(pack(local_dist[i], start_v + i));
            }
        }

        // Vòng lặp con: duyệt và relax cạnh cho các đỉnh thuộc lô hiện tại cho đến khi hội tụ
        while (true) {
            int local_size = local_active.size();
            vector<int> recvcounts(size);
            MPI_Allgather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            vector<int> displs(size);
            int total_active = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = total_active;
                total_active += recvcounts[i];
            }

            if (total_active == 0) {
                break;
            }

            vector<uint64_t> global_active(total_active);
            MPI_Allgatherv(local_active.data(), local_size, MPI_UINT64_T,
                           global_active.data(), recvcounts.data(), displs.data(), MPI_UINT64_T,
                           MPI_COMM_WORLD);

            local_active.clear();

            // Relax các cạnh của toàn bộ các đỉnh hoạt động trong cụm hiện tại
            for (int i = 0; i < total_active; ++i) {
                uint32_t dist_u, u;
                unpack(global_active[i], dist_u, u);

                if (u >= start_v && u < end_v) {
                    local_visited[u - start_v] = true;
                }

                int edge_start = p_local_head[u];
                int edge_end = p_local_head[u + 1];
                for (int e = edge_start; e < edge_end; ++e) {
                    int v = p_local_to[e];
                    int w = p_local_weight[e];
                    int local_v = v - start_v;
                    
                    uint32_t new_dist = dist_u + w;
                    if (new_dist < local_dist[local_v]) {
                        local_dist[local_v] = new_dist;
                        local_visited[local_v] = false; // Đặt lại về false để cho phép duyệt lại tối ưu hơn
                        local_pq.push(pack(new_dist, v)); 

                        // ĐÁNH DẤU TRỄ: Không push ngay vào active list, chỉ đánh dấu trạng thái hoạt động
                        if (new_dist <= threshold) {
                            in_active_list[local_v] = true;
                        }
                    }
                }
            }

            // GOM LÔ TRỄ: Duyệt qua danh sách đánh dấu để đưa đỉnh hoạt động duy nhất với khoảng cách ngắn nhất vào active list
            for (int i = 0; i < local_N; ++i) {
                if (in_active_list[i]) {
                    if (!local_visited[i]) {
                        local_active.push_back(pack(local_dist[i], start_v + i));
                    }
                    in_active_list[i] = false; // Đặt lại trạng thái cho chu kỳ sau
                }
            }
        }

        if (rank == 0 && bucket_step % 10 == 0) {
            cout << "[Master] Buoc phan lo " << bucket_step << " | Nguong khoang cach hien tai: " << threshold << "\n";
            cout.flush();
        }
        bucket_step++;
    }

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

        cout << "\n================ KET QUA DANH GIA HE CO =================\n";
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