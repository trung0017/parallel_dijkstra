#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <climits>
#include <iomanip>
#include <cstring>
#include <mpi.h>
#include <omp.h>

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

// --- PARALLEL SSSP HYBRID MPI + OPENMP (BSP-SPFA) ---
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
    #pragma omp parallel for
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

    // Mỗi tiến trình duy trì mảng khoảng cách toàn cục dist và mảng đối chiếu prev_dist kích thước V
    vector<uint32_t> dist(V, INF);
    vector<uint32_t> prev_dist(V, INF);
    
    dist[source] = 0;
    prev_dist[source] = 0;

    // Các mảng hàng đợi hoạt động cục bộ
    vector<int> active_local(V);
    vector<int> next_active_local(V);
    int active_count = 1;
    active_local[0] = source; // Tất cả tiến trình đều bắt đầu từ đỉnh nguồn

    vector<bool> in_queue(local_N, false);
    if (source >= start_v && source < end_v) {
        in_queue[source - start_v] = true;
    }

    const int* p_local_head = local_head.data();
    const int* p_local_to = local_to.data();
    const int* p_local_weight = local_weight.data();

    int sync_step = 0;

    // --- VÒNG LẶP CHÍNH BULK SYNCHRONOUS SPFA ---
    while (true) {
        // Pha 1: Duyệt cạnh cục bộ song song bằng OpenMP (Hoàn toàn không truyền tin)
        while (active_count > 0) {
            int next_active_count = 0;

            #pragma omp parallel
            {
                vector<int> private_active;

                #pragma omp for nowait
                for (int i = 0; i < active_count; ++i) {
                    int u = active_local[i];
                    int edge_start = p_local_head[u];
                    int edge_end = p_local_head[u + 1];

                    for (int e = edge_start; e < edge_end; ++e) {
                        int v = p_local_to[e];
                        int w = p_local_weight[e];
                        int local_v = v - start_v;

                        if (dist[u] + w < dist[local_v]) {
                            #pragma omp atomic write
                            dist[local_v] = dist[u] + w;

                            if (!in_queue[local_v]) {
                                in_queue[local_v] = true;
                                private_active.push_back(v);
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int v : private_active) {
                        next_active_local[next_active_count++] = v;
                    }
                }
            }

            // Giải phóng trạng thái đánh dấu hàng đợi cục bộ
            #pragma omp parallel for
            for (int i = 0; i < active_count; ++i) {
                int u = active_local[i];
                if (u >= start_v && u < end_v) {
                    in_queue[u - start_v] = false;
                }
            }

            active_local.swap(next_active_local);
            active_count = next_active_count;
        }

        // Pha 2: Đồng bộ hóa toàn bộ mảng khoảng cách dist trên toàn cụm máy tính
        MPI_Allreduce(MPI_IN_PLACE, dist.data(), V, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);

        // Pha 3: Kiểm tra các cập nhật toàn cục mới và nạp vào active_local cho pha sau
        active_count = 0;
        #pragma omp parallel
        {
            vector<int> private_active;
            #pragma omp for nowait
            for (int u = 0; u < V; ++u) {
                if (dist[u] < prev_dist[u]) {
                    prev_dist[u] = dist[u];
                    private_active.push_back(u);
                    
                    if (u >= start_v && u < end_v) {
                        in_queue[u - start_v] = true;
                    }
                }
            }

            #pragma omp critical
            {
                for (int u : private_active) {
                    active_local[active_count++] = u;
                }
            }
        }

        // Kiểm tra xem toàn hệ thống đã hội tụ hoàn toàn hay chưa
        int local_active_flag = (active_count > 0) ? 1 : 0;
        int global_active_flag;
        MPI_Allreduce(&local_active_flag, &global_active_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_active_flag == 0) {
            break;
        }

        if (rank == 0 && sync_step % 10 == 0) {
            cout << "[Master] Buoc dong bo " << sync_step << "\n";
            cout.flush();
        }
        sync_step++;
    }

    // Thu thập kết quả mảng khoảng cách cục bộ để chuẩn bị Gatherv
    #pragma omp parallel for
    for (int i = 0; i < local_N; ++i) {
        local_dist[i] = dist[start_v + i];
    }

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
        cout << " KHAO SAT DIJKSTRA SONG SONG HYBRID MPI + OPENMP (BSP-SPFA)\n";
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