#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <mpi.h>
#include <omp.h>

#define INF 2000000000

// Cấu trúc đồ thị nén CSR (Compressed Sparse Row)
typedef struct {
    int* row_ptr;  // Kích thước: local_N + 1
    int* col_idx;  // Kích thước: local_N * 10
    int* weight;   // Kích thước: local_N * 10
} CSRGraph;

// Bộ sinh số ngẫu nhiên Xorshift64 siêu nhanh, luồng an toàn (thread-safe)
inline uint32_t fast_rand(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}

int main(int argc, char** argv) {
    // Khởi tạo MPI hỗ trợ đa luồng (MPI_THREAD_FUNNELED)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10000000; // Quy mô đồ thị: 10 triệu đỉnh
    int src = 0;      // Đỉnh nguồn

    // Phân chia tập đỉnh cho từng Process (Row-wise Partitioning)
    int local_N = N / size;
    int local_start = rank * local_N;
    int local_end = (rank == size - 1) ? N : local_start + local_N;
    int current_local_N = local_end - local_start;

    if (rank == 0) {
        printf("========================================================\n");
        cout << " KHAO SAT SSSP HYBRID MPI + OPENMP CHUAN (BSP-SPFA)\n";
        printf(" Quy mo do thi: %d dinh | 100 trieu canh\n", N);
        printf(" So luong tien trinh (MPI): %d | Luong OpenMP: %d\n", size, omp_get_max_threads());
        printf("========================================================\n");
        printf("[Master] Dang tu dong khoi tao do thi CSR cuc bo song song...\n");
        fflush(stdout);
    }

    // --- KHỞI TẠO ĐỒ THỊ CSR CỤC BỘ KHÔNG CHẶN (Bản song song hóa hoàn toàn) ---
    CSRGraph local_g;
    local_g.row_ptr = (int*)malloc((current_local_N + 1) * sizeof(int));
    local_g.col_idx = (int*)malloc(current_local_N * 10 * sizeof(int));
    local_g.weight = (int*)malloc(current_local_N * 10 * sizeof(int));

    auto t_gen_start = chrono::high_resolution_clock::now();
    
    // Song song hóa quá trình sinh đồ thị cục bộ bằng OpenMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < current_local_N; i++) {
            int u = local_start + i;
            uint64_t state = (uint64_t)u + 42 + tid; // Seed luồng độc lập
            local_g.row_ptr[i] = i * 10;
            
            for (int j = 0; j < 10; j++) {
                uint32_t r_val = fast_rand(&state);
                local_g.col_idx[i * 10 + j] = r_val % N;
                local_g.weight[i * 10 + j] = 1 + (r_val % 100);
            }
        }
    }
    local_g.row_ptr[current_local_N] = current_local_N * 10;

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_gen_end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        double gen_time = chrono::duration<double>(t_gen_end - t_gen_start).count();
        printf("[Master] Khoi tao CSR cuc bo hoan tat trong: %.4f giay.\n", gen_time);
        printf("[Master] Dang thuc thi phien ban song song hoa BSP-SPFA...\n");
        fflush(stdout);
    }

    auto t_par_start = chrono::high_resolution_clock::now();

    // Cấp phát mảng khoảng cách cục bộ và mảng khoảng cách truớc đó để đối chiếu thay đổi
    int* dist = (int*)malloc(N * sizeof(int));
    int* prev_dist = (int*)malloc(current_local_N * sizeof(int));
    char* in_queue = (char*)calloc(current_local_N, sizeof(char));

    // Khởi tạo mảng khoảng cách ban đầu song song bằng OpenMP
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        dist[i] = INF;
    }
    dist[src] = 0;

    #pragma omp parallel for
    for (int i = 0; i < current_local_N; i++) {
        prev_dist[i] = INF;
    }
    if (src >= local_start && src < local_end) {
        prev_dist[src - local_start] = 0;
    }

    // Các mảng đệm hoạt động phục vụ duyệt song song cục bộ
    int* active_local = (int*)malloc(current_local_N * sizeof(int));
    int* next_active_local = (int*)malloc(current_local_N * sizeof(int));
    int active_count = 0;

    if (src >= local_start && src < local_end) {
        active_local[0] = src;
        in_queue[source - local_start] = 1;
        active_count = 1;
    }

    int sync_step = 0;

    // --- VÒNG LẶP CHÍNH BULK SYNCHRONOUS SPFA ---
    while (true) {
        // Pha 1: Duyệt cạnh cục bộ song song cho đến khi hội tụ hoàn toàn (Hoàn toàn không truyền tin)
        while (active_count > 0) {
            int next_active_count = 0;

            #pragma omp parallel
            {
                // Bộ nhớ đệm luồng cục bộ để giảm thiểu xung đột khóa tranh chấp ghi dữ liệu
                vector<int> private_active;
                
                #pragma omp for nowait
                for (int i = 0; i < active_count; ++i) {
                    int u = active_local[i];
                    int local_u_idx = u - local_start;
                    int edge_start = local_g.row_ptr[local_u_idx];
                    int edge_end = local_g.row_ptr[local_u_idx + 1];

                    for (int e = edge_start; e < edge_end; ++e) {
                        int v = local_g.col_idx[e];
                        int w = local_g.weight[e];

                        if (dist[u] + w < dist[v]) {
                            #pragma omp atomic write
                            dist[v] = dist[u] + w;

                            // Nếu đỉnh kề thuộc phân vùng cục bộ, đưa vào mảng đệm cục bộ cho chu kỳ sau
                            if (v >= local_start && v < local_end) {
                                int local_v_idx = v - local_start;
                                if (!in_queue[local_v_idx]) {
                                    in_queue[local_v_idx] = 1;
                                    private_active.push_back(v);
                                }
                            }
                        }
                    }
                }

                // Gộp kết quả hoạt động từ các luồng vào mảng hoạt động chung cục bộ
                #pragma omp critical
                {
                    for (int v : private_active) {
                        next_active_local[next_active_count++] = v;
                    }
                }
            }

            // Giải phóng đánh dấu trạng thái hoạt động của các đỉnh vừa duyệt
            #pragma omp parallel for
            for (int i = 0; i < active_count; ++i) {
                int local_u_idx = active_local[i] - local_start;
                in_queue[local_u_idx] = 0;
            }

            // Hoán đổi con trỏ mảng đệm
            int* temp_ptr = active_local;
            active_local = next_active_local;
            next_active_local = temp_ptr;
            active_count = next_active_count;
        }

        // Pha 2: Đồng bộ hóa định kỳ mảng khoảng cách tại chỗ giữa các Processes dùng MPI_IN_PLACE
        MPI_Allreduce(MPI_IN_PLACE, dist, N, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Pha 3: Kiểm tra các cập nhật từ các máy khác để nạp lại vào hàng đợi hoạt động cục bộ
        active_count = 0;
        #pragma omp parallel for reduction(+:active_count)
        for (int i = 0; i < current_local_N; ++i) {
            int u = local_start + i;
            if (dist[u] < prev_dist[i]) {
                prev_dist[i] = dist[u];
                active_local[active_count++] = u;
                in_queue[i] = 1;
            }
        }

        // Thuật toán hội tụ khi không còn tiến trình nào trong toàn hệ thống phát sinh cập nhật mới
        int local_active_flag = (active_count > 0) ? 1 : 0;
        int global_active_flag;
        MPI_Allreduce(&local_active_flag, &global_active_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_active_flag == 0) {
            break; 
        }

        if (rank == 0 && sync_step % 10 == 0) {
            printf("[Master] Buoc dong bo %d hoan tat.\n", sync_step);
            fflush(stdout);
        }
        sync_step++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_par_end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        double par_time = chrono::duration<double>(t_par_end - t_par_start).count();
        printf("[Master] Hoan thanh SSSP song song trong: %.4f giay.\n", par_time);
        printf("[Master] Khoang cach tu 0 -> %d la: %d\n", N - 1, dist[N - 1]);
        printf("========================================================\n");
        fflush(stdout);
    }

    // Giải phóng tài nguyên cục bộ
    free(local_g.row_ptr);
    free(local_g.col_idx);
    free(local_g.weight);
    free(dist);
    free(prev_dist);
    free(in_queue);
    free(active_local);
    free(next_active_local);

    MPI_Finalize();
    return 0;
}