#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100  // Размер матрицы

void print_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
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

    int local_rows_count = N / size + (rank < N % size ? 1 : 0);
    local_rows = (double*)malloc(local_rows_count * N * sizeof(double));

    if (rank == 0) {
        matrix = (double*)malloc(N * N * sizeof(double));
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = rand() % 10 + 1;
            }
        }
    }

    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (N / size + (i < N % size ? 1 : 0)) * N;
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, local_rows, local_rows_count * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; i++) {
        int root = i % size;
        int root_local_row = i / size;
        double* root_row = (double*)malloc(N * sizeof(double));

        if (rank == root) {
            for(int j = 0; j < N; j++){
                root_row[j] = local_rows[root_local_row * N + j];
            }
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
        printf("Determinant: %.3f\n", determinant);
    }

    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Elapsed time: %.6f seconds\n", end_time - start_time);
    }

    free(local_rows);
    if(rank == 0){
        free(matrix);
    }
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}