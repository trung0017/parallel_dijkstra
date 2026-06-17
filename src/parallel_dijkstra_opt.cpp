#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include <iomanip>

using namespace std;

// Sử dụng double để tương thích với MPI_DOUBLE_INT
const double INF = 1e18; 
const int V = 100000;      // Giảm xuống để chạy nhanh trên GitHub Actions
const int E = 1000000;     
const int SOURCE_NODE = 0;

struct Edge {
    int to;
    double weight;
};

// Struct bắt buộc phải là {double, int} để dùng MPI_DOUBLE_INT
struct DistRank {
    double dist;
    int rank;
};

void generateGraph(vector<vector<Edge>>& adj) {
    adj.assign(V, vector<Edge>());
    mt19937 gen(42);
    uniform_int_distribution<int> dist_v(0, V - 1);
    uniform_real_distribution<double> dist_w(1.0, 100.0);

    for (int i = 0; i < E; ++i) {
        int u = dist_v(gen);
        int v = dist_v(gen);
        double w = dist_w(gen);
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

    vector<vector<Edge>> adj;
    generateGraph(adj);

    int local_n = V / size;
    int remainder = V % size;
    int start_v = rank * local_n + min(rank, remainder);
    int end_v = start_v + local_n + (rank < remainder ? 1 : 0);

    vector<double> dists(V, INF);
    vector<bool> visited(V, false);
    
    // Priority queue để tìm min local
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    if (SOURCE_NODE >= start_v && SOURCE_NODE < end_v) {
        dists[SOURCE_NODE] = 0;
        pq.push({0.0, SOURCE_NODE});
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = chrono::high_resolution_clock::now();

    // Vòng lặp chính: Tối đa V lần
    for (int count = 0; count < V; ++count) {
        DistRank local_min = {INF, rank};
        
        // Lấy đỉnh có dist nhỏ nhất chưa visited trong vùng quản lý
        while (!pq.empty()) {
            int u = pq.top().second;
            double d = pq.top().first;
            if (visited[u]) {
                pq.pop();
                continue;
            }
            local_min.dist = d;
            break;
        }

        DistRank global_min;
        // Tìm min trên toàn bộ các process
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global_min.dist >= INF) break;

        // Xác định ID của đỉnh u global
        int u_global;
        if (rank == global_min.rank) {
            u_global = pq.top().second;
            pq.pop();
        }
        // Gửi ID đỉnh u_global cho tất cả các process khác
        MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
        
        visited[u_global] = true;

        // Relax các cạnh của đỉnh u_global
        for (auto& edge : adj[u_global]) {
            int v = edge.to;
            // Chỉ cập nhật nếu đỉnh v thuộc quyền quản lý của rank này
            if (v >= start_v && v < end_v) {
                if (!visited[v] && global_min.dist + edge.weight < dists[v]) {
                    dists[v] = global_min.dist + edge.weight;
                    pq.push({dists[v], v});
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