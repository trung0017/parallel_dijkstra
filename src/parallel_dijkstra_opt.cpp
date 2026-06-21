// #include <mpi.h>
// #include <iostream>
// #include <vector>
// #include <queue>
// #include <limits>
// #include <chrono>
// #include <random>
// #include <iomanip>
// #include <cmath>

// using namespace std;

// const double INF = 1e18;
// const int V = 50000;       // Giảm V xuống một chút để thấy rõ Speedup trên 1 máy
// const int E = 500000;
// const int SOURCE_NODE = 0;

// struct Edge {
//     int to;
//     double weight;
// };

// struct DistRank {
//     double dist;
//     int rank;
// };

// // Hàm sinh đồ thị (đảm bảo mọi rank sinh ra cùng một đồ thị nhờ seed 42)
// void generateGraph(vector<vector<Edge>>& adj) {
//     adj.assign(V, vector<Edge>());
//     mt19937 gen(42);
//     uniform_int_distribution<int> dist_v(0, V - 1);
//     uniform_real_distribution<double> dist_w(1.0, 10.0);

//     for (int i = 0; i < E; ++i) {
//         int u = dist_v(gen);
//         int v = dist_v(gen);
//         double w = dist_w(gen);
//         if (u != v) {
//             adj[u].push_back({v, w});
//         }
//     }
// }

// // Thuật toán Dijkstra tuần tự
// vector<double> dijkstraSequential(const vector<vector<Edge>>& adj) {
//     vector<double> dists(V, INF);
//     dists[SOURCE_NODE] = 0;
//     priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
//     pq.push({0.0, SOURCE_NODE});

//     while (!pq.empty()) {
//         double d = pq.top().first;
//         int u = pq.top().second;
//         pq.pop();

//         if (d > dists[u]) continue;

//         for (auto& edge : adj[u]) {
//             if (dists[u] + edge.weight < dists[edge.to]) {
//                 dists[edge.to] = dists[u] + edge.weight;
//                 pq.push({dists[edge.to], edge.to});
//             }
//         }
//     }
//     return dists;
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     vector<vector<Edge>> adj;
//     generateGraph(adj);

//     // --- PHẦN 1: CHẠY TUẦN TỰ (CHỈ RANK 0) ---
//     vector<double> seq_res;
//     double seq_time = 0;
//     if (rank == 0) {
//         cout << "--- SO SANH DIJKSTRA TUAN TU VS SONG SONG ---" << endl;
//         cout << "V = " << V << ", E = " << E << ", Processes = " << size << endl;
        
//         auto s_start = chrono::high_resolution_clock::now();
//         seq_res = dijkstraSequential(adj);
//         auto s_end = chrono::high_resolution_clock::now();
//         seq_time = chrono::duration<double>(s_end - s_start).count();
//         cout << "Thoi gian tuan tu: " << fixed << setprecision(4) << seq_time << " giay" << endl;
//     }

//     // --- PHẦN 2: CHẠY SONG SONG (MPI) ---
//     int local_n = V / size;
//     int remainder = V % size;
//     int start_v = rank * local_n + min(rank, remainder);
//     int end_v = start_v + local_n + (rank < remainder ? 1 : 0);
//     int local_count = end_v - start_v;

//     vector<double> local_dists(V, INF);
//     vector<bool> local_visited(V, false);
//     priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

//     if (SOURCE_NODE >= start_v && SOURCE_NODE < end_v) {
//         local_dists[SOURCE_NODE] = 0;
//         pq.push({0.0, SOURCE_NODE});
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto p_start = chrono::high_resolution_clock::now();

//     for (int count = 0; count < V; ++count) {
//         DistRank local_min = {INF, rank};
//         while (!pq.empty()) {
//             int u = pq.top().second;
//             if (local_visited[u]) { pq.pop(); continue; }
//             local_min.dist = pq.top().first;
//             break;
//         }

//         DistRank global_min;
//         MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

//         if (global_min.dist >= INF) break;

//         int u_global;
//         if (rank == global_min.rank) {
//             u_global = pq.top().second;
//             pq.pop();
//         }
//         MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
//         local_visited[u_global] = true;

//         for (auto& edge : adj[u_global]) {
//             int v = edge.to;
//             if (v >= start_v && v < end_v) {
//                 if (!local_visited[v] && global_min.dist + edge.weight < local_dists[v]) {
//                     local_dists[v] = global_min.dist + edge.weight;
//                     pq.push({local_dists[v], v});
//                 }
//             }
//         }
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto p_end = chrono::high_resolution_clock::now();
//     double par_time = chrono::duration<double>(p_end - p_start).count();

//     // --- PHẦN 3: THU THẬP KẾT QUẢ VÀ IN ---
//     // Gom các đoạn local_dists về Rank 0
//     vector<double> final_par_dist;
//     if (rank == 0) final_par_dist.resize(V);

//     // Tính toán lại counts và displacements cho Gatherv
//     vector<int> counts(size), displs(size);
//     for (int i = 0; i < size; ++i) {
//         counts[i] = V / size + (i < remainder ? 1 : 0);
//         displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
//     }

//     MPI_Gatherv(&local_dists[start_v], local_count, MPI_DOUBLE,
//                 rank == 0 ? final_par_dist.data() : nullptr,
//                 counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         cout << "Thoi gian song song: " << par_time << " giay" << endl;
//         cout << "Speedup: " << seq_time / par_time << "x" << endl;

//         // In 10 kết quả đầu tiên để kiểm tra
//         cout << "\nKiem tra 10 dinh dau tien (Node: Tuan tu | Song song):" << endl;
//         bool match = true;
//         for (int i = 0; i < min(10, V); ++i) {
//             cout << "Node " << i << ": " << setw(8) << seq_res[i] << " | " << setw(8) << final_par_dist[i] << endl;
//             if (abs(seq_res[i] - final_par_dist[i]) > 1e-6) match = false;
//         }
        
//         if (match) cout << "\n=> KET QUA CHINH XAC (Khap voi ban tuan tu)!" << endl;
//         else cout << "\n=> KET QUA SAI LECH!" << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }

// #include <mpi.h>
// #include <iostream>
// #include <vector>
// #include <queue>
// #include <limits>
// #include <chrono>
// #include <random>
// #include <iomanip>
// #include <cmath>

// using namespace std;

// const double INF = 1e18;
// const int V = 5000000;  // Có thể tăng lên khi chạy trên máy mạnh
// const int E = 50000000;
// const int SOURCE_NODE = 0;

// struct Edge {
//     int to;
//     double weight;
// };

// struct DistRank {
//     double dist;
//     int rank;
// };

// // Cấu trúc tin nhắn để truyền thông tin cập nhật cạnh qua mạng
// struct UpdateMsg {
//     int v_to;
//     double new_dist;
// };

// // --- HÀM DIJKSTRA TUẦN TỰ ---
// vector<double> dijkstraSequential(int v_count, const vector<vector<Edge>>& adj) {
//     vector<double> dists(v_count, INF);
//     dists[SOURCE_NODE] = 0;
//     priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
//     pq.push({0.0, SOURCE_NODE});

//     while (!pq.empty()) {
//         double d = pq.top().first;
//         int u = pq.top().second;
//         pq.pop();

//         if (d > dists[u]) continue;

//         for (const auto& edge : adj[u]) {
//             if (dists[u] + edge.weight < dists[edge.to]) {
//                 dists[edge.to] = dists[u] + edge.weight;
//                 pq.push({dists[edge.to], edge.to});
//             }
//         }
//     }
//     return dists;
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // 1. Phân chia vùng đỉnh quản lý (1D Partitioning)
//     int local_n = V / size;
//     int remainder = V % size;
//     int start_v = rank * local_n + min(rank, remainder);
//     int end_v = start_v + local_n + (rank < remainder ? 1 : 0);
//     int local_count = end_v - start_v;

//     // 2. Sinh đồ thị (Rank 0 sinh toàn bộ để chạy tuần tự, các Rank khác chỉ sinh phần của mình)
//     vector<vector<Edge>> full_adj;
//     vector<vector<Edge>> local_adj(V); // Chỉ chứa cạnh của các đỉnh thuộc quản lý của rank này

//     mt19937 gen(42);
//     uniform_int_distribution<int> dist_v(0, V - 1);
//     uniform_real_distribution<double> dist_w(1.0, 10.0);

//     if (rank == 0) full_adj.resize(V);

//     for (int i = 0; i < E; ++i) {
//         int u = dist_v(gen);
//         int v = dist_v(gen);
//         double w = dist_w(gen);
//         if (u == v) continue;

//         if (rank == 0) full_adj[u].push_back({v, w});
//         // Phân vùng 1D: Chỉ lưu cạnh nếu đỉnh xuất phát u thuộc rank này
//         if (u >= start_v && u < end_v) {
//             local_adj[u].push_back({v, w});
//         }
//     }

//     // --- PHẦN 1: CHẠY TUẦN TỰ (CHỈ RANK 0) ---
//     vector<double> seq_res;
//     double seq_time = 0;
//     if (rank == 0) {
//         cout << "--- DIJKSTRA MPI 1D PARTITIONING ---" << endl;
//         cout << "V = " << V << ", E = " << E << ", NP = " << size << endl;
//         auto s_start = chrono::high_resolution_clock::now();
//         seq_res = dijkstraSequential(V, full_adj);
//         auto s_end = chrono::high_resolution_clock::now();
//         seq_time = chrono::duration<double>(s_end - s_start).count();
//         cout << "Thoi gian tuan tu: " << fixed << setprecision(4) << seq_time << " giay" << endl;
//         full_adj.clear(); // Giải phóng bộ nhớ cho Rank 0
//     }

//     // --- PHẦN 2: CHẠY SONG SONG (MPI) ---
//     vector<double> local_dists(V, INF);
//     vector<bool> local_visited(V, false);
//     priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

//     if (SOURCE_NODE >= start_v && SOURCE_NODE < end_v) {
//         local_dists[SOURCE_NODE] = 0;
//         pq.push({0.0, SOURCE_NODE});
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto p_start = chrono::high_resolution_clock::now();

//     for (int count = 0; count < V; ++count) {
//         DistRank local_min = {INF, rank};
//         while (!pq.empty()) {
//             int u = pq.top().second;
//             if (local_visited[u]) { pq.pop(); continue; }
//             local_min.dist = pq.top().first;
//             break;
//         }

//         DistRank global_min;
//         MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

//         if (global_min.dist >= INF) break;

//         int u_global;
//         vector<UpdateMsg> updates;

//         // Rank nắm giữ đỉnh u_global sẽ chuẩn bị danh sách cạnh để gửi
//         if (rank == global_min.rank) {
//             u_global = pq.top().second;
//             pq.pop();
//             for (auto& edge : local_adj[u_global]) {
//                 updates.push_back({edge.to, global_min.dist + edge.weight});
//             }
//         }

//         // Thông báo cho mọi rank biết đỉnh nào vừa được chốt
//         MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
//         local_visited[u_global] = true;

//         // Broadcast số lượng cạnh và danh sách cạnh của u_global
//         int num_updates = updates.size();
//         MPI_Bcast(&num_updates, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
//         if (num_updates > 0) {
//             if (rank != global_min.rank) updates.resize(num_updates);
//             MPI_Bcast(updates.data(), num_updates * sizeof(UpdateMsg), MPI_BYTE, global_min.rank, MPI_COMM_WORLD);

//             for (const auto& up : updates) {
//                 // Chỉ rank quản lý đỉnh 'up.v_to' mới thực hiện cập nhật
//                 if (up.v_to >= start_v && up.v_to < end_v) {
//                     if (!local_visited[up.v_to] && up.new_dist < local_dists[up.v_to]) {
//                         local_dists[up.v_to] = up.new_dist;
//                         pq.push({local_dists[up.v_to], up.v_to});
//                     }
//                 }
//             }
//         }
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     auto p_end = chrono::high_resolution_clock::now();
//     double par_time = chrono::duration<double>(p_end - p_start).count();

//     // --- PHẦN 3: THU THẬP VÀ KIỂM TRA KẾT QUẢ ---
//     vector<double> final_par_dist;
//     if (rank == 0) final_par_dist.resize(V);

//     vector<int> counts(size), displs(size);
//     for (int i = 0; i < size; ++i) {
//         counts[i] = V / size + (i < (V % size) ? 1 : 0);
//         displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
//     }

//     // Thu thập phần dist mà mỗi rank quản lý về Rank 0
//     MPI_Gatherv(&local_dists[start_v], local_count, MPI_DOUBLE,
//                 rank == 0 ? final_par_dist.data() : nullptr,
//                 counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         cout << "Thoi gian song song: " << par_time << " giay" << endl;
//         cout << "Speedup: " << seq_time / par_time << "x" << endl;

//         cout << "\nKiem tra 10 dinh dau tien (Node: Tuan tu | Song song):" << endl;
//         bool match = true;
//         for (int i = 0; i < min(10, V); ++i) {
//             cout << "Node " << i << ": " << setw(8) << seq_res[i] << " | " << setw(8) << final_par_dist[i] << endl;
//             if (abs(seq_res[i] - final_par_dist[i]) > 1e-6) match = false;
//         }
//         if (match) cout << "\n=> KET QUA CHINH XAC!" << endl;
//         else cout << "\n=> KET QUA SAI LECH!" << endl;
//     }

//     MPI_Finalize();
//     return 0;
// }

#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

const double INF = 1e18;
const int V = 3000000;  // Giảm V để quan sát sự khác biệt rõ hơn
const int E = 30000000;
const int SOURCE_NODE = 0;

struct Edge {
    int to;
    double weight;
};

struct DistRank {
    double dist;
    int rank;
};

// Cấu trúc cho bản tin cập nhật
struct UpdateMsg {
    int v_to;
    double new_dist;
};

// Dijkstra tuần tự để so sánh
vector<double> dijkstraSequential(int v_count, const vector<vector<Edge>>& adj) {
    vector<double> dists(v_count, INF);
    dists[SOURCE_NODE] = 0;
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    pq.push({0.0, SOURCE_NODE});

    while (!pq.empty()) {
        double d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (d > dists[u]) continue;
        for (const auto& edge : adj[u]) {
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

    // Thiết lập lưới 2D (R x C)
    int R = sqrt(size);
    while (size % R != 0) R--; // Tìm R sao cho size chia hết cho R
    int C = size / R;
    int my_row = rank / C;
    int my_col = rank % C;

    // Chia vùng đỉnh theo hàng và cột
    auto get_range = [](int total, int parts, int idx) {
        int base = total / parts;
        int rem = total % parts;
        int start = idx * base + min(idx, rem);
        int end = start + base + (idx < rem ? 1 : 0);
        return make_pair(start, end);
    };

    auto row_range = get_range(V, R, my_row); // Dải đỉnh nguồn u
    auto col_range = get_range(V, C, my_col); // Dải đỉnh đích v

    // 1. Sinh đồ thị phân vùng 2D
    vector<vector<Edge>> full_adj;
    vector<vector<Edge>> local_adj(V); 
    mt19937 gen(42);
    uniform_int_distribution<int> dist_v(0, V - 1);
    uniform_real_distribution<double> dist_w(1.0, 10.0);

    if (rank == 0) full_adj.resize(V);

    for (int i = 0; i < E; ++i) {
        int u = dist_v(gen);
        int v = dist_v(gen);
        double w = dist_w(gen);
        if (u == v) continue;
        if (rank == 0) full_adj[u].push_back({v, w});
        
        // Cạnh u -> v chỉ lưu ở rank (r, c) nếu u thuộc RowRegion r và v thuộc ColRegion c
        if (u >= row_range.first && u < row_range.second && 
            v >= col_range.first && v < col_range.second) {
            local_adj[u].push_back({v, w});
        }
    }

    // --- CHẠY TUẦN TỰ ---
    vector<double> seq_res;
    double seq_time = 0;
    if (rank == 0) {
        cout << "--- DIJKSTRA MPI 2D PARTITIONING ---" << endl;
        cout << "Lưới: " << R << "x" << C << ", V=" << V << ", E=" << E << endl;
        auto s_start = chrono::high_resolution_clock::now();
        seq_res = dijkstraSequential(V, full_adj);
        seq_time = chrono::duration<double>(chrono::high_resolution_clock::now() - s_start).count();
        cout << "Thoi gian tuan tu: " << seq_time << " s" << endl;
    }

    // --- CHẠY SONG SONG 2D ---
    // Trong 2D, khoảng cách dist thường được quản lý bởi các cột
    vector<double> local_dists(V, INF);
    vector<bool> settled(V, false);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    // Khởi tạo điểm gốc
    if (SOURCE_NODE >= col_range.first && SOURCE_NODE < col_range.second) {
        local_dists[SOURCE_NODE] = 0;
        pq.push({0.0, SOURCE_NODE});
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto p_start = chrono::high_resolution_clock::now();

    for (int count = 0; count < V; ++count) {
        DistRank local_min = {INF, rank};
        while (!pq.empty()) {
            if (settled[pq.top().second]) { pq.pop(); continue; }
            local_min.dist = pq.top().first;
            break;
        }

        DistRank global_min;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (global_min.dist >= INF) break;

        int u_global;
        if (rank == global_min.rank) u_global = pq.top().second;
        MPI_Bcast(&u_global, 1, MPI_INT, global_min.rank, MPI_COMM_WORLD);
        settled[u_global] = true;

        // Cập nhật 2D: 
        // 1. Tìm xem ai quản lý các cạnh đi ra từ u_global (là các rank ở hàng quản lý u_global)
        int owner_row = -1;
        for(int r=0; r<R; ++r) {
            auto r_rng = get_range(V, R, r);
            if (u_global >= r_rng.first && u_global < r_rng.second) { owner_row = r; break; }
        }

        vector<UpdateMsg> updates;
        if (my_row == owner_row) {
            for (auto& edge : local_adj[u_global]) {
                updates.push_back({edge.to, global_min.dist + edge.weight});
            }
        }

        // 2. Các rank trong hàng đó gửi cập nhật cho các cột tương ứng
        // (Trong bản đơn giản này, ta vẫn dùng Bcast nhưng chỉ trong phạm vi hàng/cột nếu tối ưu)
        // Ở đây ta dùng cách tiếp cận Edge-Partitioning: các rank ở hàng owner_row xử lý local
        for (const auto& up : updates) {
            if (up.new_dist < local_dists[up.v_to]) {
                local_dists[up.v_to] = up.new_dist;
                pq.push({local_dists[up.v_to], up.v_to});
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto p_end = chrono::high_resolution_clock::now();
    double par_time = chrono::duration<double>(p_end - p_start).count();

    // Thu thập kết quả
    vector<double> final_dist(V);
    // Mỗi cột quản lý một phần kết quả
    vector<double> my_col_results;
    for(int i = col_range.first; i < col_range.second; ++i) my_col_results.push_back(local_dists[i]);

    // Gom dữ liệu từ các cột (chỉ cần lấy từ 1 hàng bất kỳ, VD hàng 0)
    if (my_row == 0) {
        // Thu thập kết quả từ các tiến trình trong cùng hàng 0
        // (Logic phức tạp hơn Gatherv thông thường vì chỉ thu thập từ một số rank)
    }

    if (rank == 0) {
        cout << "Thoi gian song song 2D: " << par_time << " s" << endl;
        cout << "Speedup: " << seq_time / par_time << "x" << endl;
    }

    MPI_Finalize();
    return 0;
}
