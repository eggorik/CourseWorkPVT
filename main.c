#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100  // Размер матрицы

void print_matrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.3f ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int rank, size;
    double* matrix = NULL;
    double* local_rows = NULL;
    double determinant = 1.0;
    double local_determinant = 1.0;
    int swap_count = 0;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();

    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = rand() % 10 + 1;
            }
        }
        // printf("Initial matrix:\n");
        // print_matrix(matrix);
    }

    local_rows = (double*)malloc((N / size + (N % size > rank)) * N * sizeof(double));

    int local_rows_count = 0;
    for (int i = rank; i < N; i += size) {
        MPI_Scatter(matrix + i * N, N, MPI_DOUBLE, local_rows + local_rows_count * N, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_rows_count++;
    }

    for (int i = 0; i < N; i++) {
        int root = i % size;
        int root_local_row = i / size;
        double* root_row = NULL;

        if (rank == root) {
            if (fabs(local_rows[root_local_row * N + i]) < 1e-10) {
                int best_row = i;
                double best_val = fabs(local_rows[root_local_row * N + i]);
                for (int k = i + 1; k < N; k++) {
                    double val = fabs(matrix[k * N + i]);
                    if (val > best_val) {
                        best_val = val;
                        best_row = k;
                    }
                }
                if (best_row != i) {
                    for (int j = 0; j < N; j++) {
                        double temp = matrix[i * N + j];
                        matrix[i * N + j] = matrix[best_row * N + j];
                        matrix[best_row * N + j] = temp;
                    }
                    swap_count++;
                }
            }
            root_row = (double*)malloc(N * sizeof(double));
            for(int j = 0; j < N; j++){
                root_row[j] = local_rows[root_local_row * N + j];
            }
        } else {
            root_row = (double*)malloc(N * sizeof(double));
        }

        MPI_Bcast(root_row, N, MPI_DOUBLE, root, MPI_COMM_WORLD);

        int local_row_index = 0;
        for (int k = rank; k < N; k += size) {
            if (k > i) {
                double factor = local_rows[local_row_index * N + i] / root_row[i];
                for (int j = i + 1; j < N; j++) {
                    local_rows[local_row_index * N + j] -= factor * root_row[j];
                }
            }
            local_row_index++;
        }
        free(root_row);
    }

    local_determinant = 1.0;
    for (int i = 0; i < local_rows_count; i++) {
        local_determinant *= local_rows[i * N + (rank + i * size)];
    }

    MPI_Reduce(&local_determinant, &determinant, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (swap_count % 2 != 0) {
            determinant = -determinant;
        }
        // printf("Determinant: %.3f\n", determinant);
        // printf("\nResult matrix:\n");
        // print_matrix(matrix);
    }

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Elapsed time: %.6f seconds\n", end_time - start_time);
    }

    free(local_rows);
    if(rank == 0){
        free(matrix);
    }

    MPI_Finalize();
    return 0;
}