#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-12

typedef struct {
    double value;
    int row;
} PivotCandidate;

static void get_row_range(int rank, int size, int n, int *start, int *count) {
    int base = n / size;
    int rem = n % size;

    if (rank < rem) {
        *count = base + 1;
        *start = rank * (*count);
    } else {
        *count = base;
        *start = rem * (base + 1) + (rank - rem) * base;
    }
}

static int owner_of_row(int row, int size, int n) {
    int base = n / size;
    int rem = n % size;

    int threshold = rem * (base + 1);

    if (row < threshold) {
        return row / (base + 1);
    }

    return rem + (row - threshold) / base;
}

static void swap_local_rows(double *matrix, int cols, int local_i, int local_j) {
    for (int j = 0; j < cols; j++) {
        double tmp = matrix[local_i * cols + j];
        matrix[local_i * cols + j] = matrix[local_j * cols + j];
        matrix[local_j * cols + j] = tmp;
    }
}

static double *read_augmented_matrix(const char *filename, int *n) {
    FILE *file = fopen(filename, "r");

    if (!file) {
        fprintf(stderr, "Erreur : impossible d'ouvrir le fichier %s\n", filename);
        return NULL;
    }

    if (fscanf(file, "%d", n) != 1 || *n <= 0) {
        fprintf(stderr, "Erreur : format invalide pour la taille de la matrice.\n");
        fclose(file);
        return NULL;
    }

    int cols = *n + 1;
    double *matrix = malloc((size_t)(*n) * cols * sizeof(double));

    if (!matrix) {
        fprintf(stderr, "Erreur : allocation mémoire impossible.\n");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *n; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &matrix[i * cols + j]) != 1) {
                fprintf(stderr, "Erreur : données manquantes dans le fichier.\n");
                free(matrix);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return matrix;
}

static void print_solution(const double *x, int n) {
    printf("\nSolution du système :\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.10f\n", i, x[i]);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage : mpirun -np <P> ./gauss_jordan_mpi <fichier_matrice>\n");
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = 0;
    double *global_matrix = NULL;

    if (rank == 0) {
        global_matrix = read_augmented_matrix(argv[1], &n);

        if (!global_matrix) {
            n = -1;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int cols = n + 1;

    int *send_counts = malloc((size_t)size * sizeof(int));
    int *displacements = malloc((size_t)size * sizeof(int));
    int *row_counts = malloc((size_t)size * sizeof(int));
    int *row_displacements = malloc((size_t)size * sizeof(int));

    if (!send_counts || !displacements || !row_counts || !row_displacements) {
        fprintf(stderr, "Rank %d : erreur allocation metadata.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int r = 0; r < size; r++) {
        int start;
        int count;

        get_row_range(r, size, n, &start, &count);

        row_counts[r] = count;
        row_displacements[r] = start;

        send_counts[r] = count * cols;
        displacements[r] = start * cols;
    }

    int local_start;
    int local_rows;

    get_row_range(rank, size, n, &local_start, &local_rows);

    double *local_matrix = calloc((size_t)local_rows * cols, sizeof(double));
    double *pivot_row = malloc((size_t)cols * sizeof(double));

    if (!local_matrix || !pivot_row) {
        fprintf(stderr, "Rank %d : erreur allocation matrice locale.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatterv(
        global_matrix,
        send_counts,
        displacements,
        MPI_DOUBLE,
        local_matrix,
        local_rows * cols,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        free(global_matrix);
        global_matrix = NULL;
    }

    for (int k = 0; k < n; k++) {
        PivotCandidate local_candidate;
        local_candidate.value = -1.0;
        local_candidate.row = -1;

        for (int i = 0; i < local_rows; i++) {
            int global_i = local_start + i;

            if (global_i >= k) {
                double value = fabs(local_matrix[i * cols + k]);

                if (value > local_candidate.value) {
                    local_candidate.value = value;
                    local_candidate.row = global_i;
                }
            }
        }

        PivotCandidate global_candidate;

        MPI_Allreduce(
            &local_candidate,
            &global_candidate,
            1,
            MPI_DOUBLE_INT,
            MPI_MAXLOC,
            MPI_COMM_WORLD
        );

        if (global_candidate.value < EPSILON) {
            if (rank == 0) {
                fprintf(stderr, "Erreur : matrice singulière ou pivot nul à l'étape %d.\n", k);
            }

            free(local_matrix);
            free(pivot_row);
            free(send_counts);
            free(displacements);
            free(row_counts);
            free(row_displacements);

            MPI_Finalize();
            return EXIT_FAILURE;
        }

        int pivot_global_row = global_candidate.row;
        int owner_k = owner_of_row(k, size, n);
        int owner_pivot = owner_of_row(pivot_global_row, size, n);

        if (pivot_global_row != k) {
            if (owner_k == owner_pivot) {
                if (rank == owner_k) {
                    int local_k = k - local_start;
                    int local_pivot = pivot_global_row - local_start;

                    swap_local_rows(local_matrix, cols, local_k, local_pivot);
                }
            } else {
                if (rank == owner_k) {
                    int local_k = k - local_start;

                    MPI_Sendrecv_replace(
                        &local_matrix[local_k * cols],
                        cols,
                        MPI_DOUBLE,
                        owner_pivot,
                        0,
                        owner_pivot,
                        0,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );
                } else if (rank == owner_pivot) {
                    int local_pivot = pivot_global_row - local_start;

                    MPI_Sendrecv_replace(
                        &local_matrix[local_pivot * cols],
                        cols,
                        MPI_DOUBLE,
                        owner_k,
                        0,
                        owner_k,
                        0,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );
                }
            }
        }

        if (rank == owner_k) {
            int local_k = k - local_start;
            double pivot = local_matrix[local_k * cols + k];

            for (int j = 0; j < cols; j++) {
                local_matrix[local_k * cols + j] /= pivot;
                pivot_row[j] = local_matrix[local_k * cols + j];
            }
        }

        MPI_Bcast(pivot_row, cols, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; i++) {
            int global_i = local_start + i;

            if (global_i == k) {
                continue;
            }

            double factor = local_matrix[i * cols + k];

            if (fabs(factor) > EPSILON) {
                for (int j = 0; j < cols; j++) {
                    local_matrix[i * cols + j] -= factor * pivot_row[j];
                }

                local_matrix[i * cols + k] = 0.0;
            }
        }
    }

    double *local_solution = malloc((size_t)local_rows * sizeof(double));

    if (!local_solution) {
        fprintf(stderr, "Rank %d : erreur allocation solution locale.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < local_rows; i++) {
        local_solution[i] = local_matrix[i * cols + n];
    }

    double *solution = NULL;

    if (rank == 0) {
        solution = malloc((size_t)n * sizeof(double));

        if (!solution) {
            fprintf(stderr, "Erreur allocation solution finale.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gatherv(
        local_solution,
        local_rows,
        MPI_DOUBLE,
        solution,
        row_counts,
        row_displacements,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        print_solution(solution, n);
        free(solution);
    }

    free(local_solution);
    free(local_matrix);
    free(pivot_row);
    free(send_counts);
    free(displacements);
    free(row_counts);
    free(row_displacements);

    MPI_Finalize();

    return EXIT_SUCCESS;
}