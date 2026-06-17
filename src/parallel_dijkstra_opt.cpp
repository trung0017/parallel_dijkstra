#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;

const double INF = 1e18;
const int V = 50000;       // Giảm V xuống một chút để thấy rõ Speedup trên 1 máy
const int E = 500000;
const int SOURCE_NODE = 0;

struct Edge {
    int to;
    double weight;
};

struct DistRank {
    double dist;
    int rank;
};

// Hàm sinh đồ thị (đảm bảo mọi rank sinh ra cùng một đồ thị nhờ seed 42)
void generateGraph(vector<vector<Edge>>& adj) {
    adj.assign(V, vector<Edge>());
    mt19937 gen(42);
    uniform_int_distribution<int> dist_v(0, V - 1);
    uniform_real_distribution<double> dist_w(1.0, 10.0);

    for (int i = 0; i < E; ++i) {
        int u = dist_v(gen);
        int v = dist_v(gen);
        double w = dist_w(gen);
        if (u != v) {
            adj[u].push_back({v, w});
        }
    }
}

// Thuật toán Dijkstra tuần tự
vector<double> dijkstraSequential(const vector<vector<Edge>>& adj) {
    vector<double> dists(V, INF);
    dists[SOURCE_NODE] = 0;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    pq.push({0.0, SOURCE_NODE});

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > dists[u]) continue;

        for (auto& edge : adj[u]) {
            if (dists[u] + edge.weight < dists[edge.to]) {
                dists[edge.to] = dists[u] + edge.weight;
                pq.push({dists[edge.to], edge.to});
            }
        }
    }
    return dists;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<Edge>> adj;
    generateGraph(adj);

    // --- PHẦN 1: CHẠY TUẦN TỰ (CHỈ RANK 0) ---
    vector<double> seq_res;
    double seq_time = 0;
    if (rank == 0) {
        cout << "--- SO SANH DIJKSTRA TUAN TU VS SONG SONG ---" << endl;
        cout << "V = " << V << ", E = " << E << ", Processes = " << size << endl;
        
        auto s_start = chrono::high_resolution_clock::now();
        seq_res = dijkstraSequential(adj);
        auto s_end = chrono::high_resolution_clock::now();
        seq_time = chrono::duration<double>(s_end - s_start).count();
        cout << "Thoi gian tuan tu: " << fixed << setprecision(4) << seq_time << " giay" << endl;
    }

    // --- PHẦN 2: CHẠY SONG SONG (MPI) ---
    int local_n = V / size;
    int remainder = V % size;
    int start_v = rank * local_n + min(rank, remainder);
    int end_v = start_v + local_n + (rank < remainder ? 1 : 0);
    int local_count = end_v - start_v;

    vector<double> local_dists(V, INF);
    vector<bool> local_visited(V, false);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    if (SOURCE_NODE >= start_v && SOURCE_NODE < end_v) {
        local_dists[SOURCE_NODE] = 0;
        pq.push({0.0, SOURCE_NODE});
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto p_start = chrono::high_resolution_clock::now();

    for (int count = 0; count < V; ++count) {
        DistRank local_min = {INF, rank};
        while (!pq.empty()) {
            int u = pq.top().second;
            if (local_visited[u]) { pq.pop(); continue; }
            local_min.dist = pq.top().first;
            break;
        }

        DistRank global_min;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global_min.dist >= INF) break;

        int u_global;
        if (rank == global_min.rank) {
            u_global = pq.top().second;
            pq.pop();
        }
        MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
        local_visited[u_global] = true;

        for (auto& edge : adj[u_global]) {
            int v = edge.to;
            if (v >= start_v && v < end_v) {
                if (!local_visited[v] && global_min.dist + edge.weight < local_dists[v]) {
                    local_dists[v] = global_min.dist + edge.weight;
                    pq.push({local_dists[v], v});
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto p_end = chrono::high_resolution_clock::now();
    double par_time = chrono::duration<double>(p_end - p_start).count();

    // --- PHẦN 3: THU THẬP KẾT QUẢ VÀ IN ---
    // Gom các đoạn local_dists về Rank 0
    vector<double> final_par_dist;
    if (rank == 0) final_par_dist.resize(V);

    // Tính toán lại counts và displacements cho Gatherv
    vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        counts[i] = V / size + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    MPI_Gatherv(&local_dists[start_v], local_count, MPI_DOUBLE,
                rank == 0 ? final_par_dist.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Thoi gian song song: " << par_time << " giay" << endl;
        cout << "Speedup: " << seq_time / par_time << "x" << endl;

        // In 10 kết quả đầu tiên để kiểm tra
        cout << "\nKiem tra 10 dinh dau tien (Node: Tuan tu | Song song):" << endl;
        bool match = true;
        for (int i = 0; i < min(10, V); ++i) {
            cout << "Node " << i << ": " << setw(8) << seq_res[i] << " | " << setw(8) << final_par_dist[i] << endl;
            if (abs(seq_res[i] - final_par_dist[i]) > 1e-6) match = false;
        }
        
        if (match) cout << "\n=> KET QUA CHINH XAC (Khap voi ban tuan tu)!" << endl;
        else cout << "\n=> KET QUA SAI LECH!" << endl;
    }

    MPI_Finalize();
    return 0;
}