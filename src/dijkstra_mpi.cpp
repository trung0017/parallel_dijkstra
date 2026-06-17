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
const int V = 1000000;      // Giảm xuống 1 triệu để test, 10 triệu cần RAM rất lớn
const int E = 10000000;     
const int SOURCE_NODE = 0;

struct Edge {
    int to;
    long long weight;
};

// Cấu trúc để dùng với MPI_MINLOC
struct {
    long long dist;
    int rank;
} local_min, global_min;

void generateGraph(vector<vector<Edge>>& adj, int v_count, int e_count) {
    adj.resize(v_count);
    mt19937 gen(42);
    uniform_int_distribution<int> dist_v(0, v_count - 1);
    uniform_int_distribution<long long> dist_w(1, 100);

    for (int i = 0; i < e_count; ++i) {
        int u = dist_v(gen);
        int v = dist_v(gen);
        long long w = dist_w(gen);
        if (u != v) {
            adj[u].push_back({v, w});
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) cout << "--- DIJKSTRA MPI OPTIMIZED ---" << endl;

    // 1. Khởi tạo đồ thị
    vector<vector<Edge>> adj;
    generateGraph(adj, V, E);

    // 2. Chia vùng quản lý đỉnh
    int local_n = V / size;
    int remainder = V % size;
    int start_v = rank * local_n + min(rank, remainder);
    int end_v = start_v + (V / size) + (rank < remainder ? 1 : 0);

    vector<long long> local_dist(V, INF); // Lưu khoảng cách tới mọi đỉnh nhưng chỉ cập nhật vùng mình quản lý
    vector<bool> local_visited(V, false);
    
    // Priority Queue để tìm min cục bộ nhanh: {distance, vertex_id}
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;

    if (SOURCE_NODE >= start_v && SOURCE_NODE < end_v) {
        local_dist[SOURCE_NODE] = 0;
        pq.push({0, SOURCE_NODE});
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = chrono::high_resolution_clock::now();

    // 3. Vòng lặp chính
    while (true) {
        // Tìm đỉnh có khoảng cách nhỏ nhất trong vùng quản lý của process này
        local_min.dist = INF;
        local_min.rank = rank;
        
        while (!pq.empty()) {
            int u = pq.top().second;
            long long d = pq.top().first;
            if (d > local_dist[u] || local_visited[u]) {
                pq.pop();
                continue;
            }
            local_min.dist = d;
            local_min.rank = rank; // Gắn rank để biết ai đang giữ min này
            break;
        }

        // Tìm min trên toàn bộ các process
        // MPI_MINLOC trả về giá trị nhỏ nhất và rank của process chứa giá trị đó
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global_min.dist == INF) break;

        // Xác định đỉnh u được chọn toàn cục
        int u_global;
        if (rank == global_min.rank) {
            u_global = pq.top().second;
            pq.pop();
            local_visited[u_global] = true;
        }
        // Broadcast đỉnh u_global để mọi người cùng biết đỉnh nào vừa được "chốt"
        MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
        local_visited[u_global] = true;

        // 4. Relax các cạnh của đỉnh u_global
        for (auto& edge : adj[u_global]) {
            int v = edge.to;
            // Chỉ process quản lý đỉnh v mới được cập nhật
            if (v >= start_v && v < end_v) {
                if (!local_visited[v] && global_min.dist + edge.weight < local_dist[v]) {
                    local_dist[v] = global_min.dist + edge.weight;
                    pq.push({local_dist[v], v});
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = chrono::high_resolution_clock::now();

    if (rank == 0) {
        chrono::duration<double> diff = end_time - start_time;
        cout << "Thoi gian thuc thi: " << fixed << setprecision(4) << diff.count() << " giay" << endl;
    }

    MPI_Finalize();
    return 0;
}