#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int n;
    int *array = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process takes input
    if (rank == 0)
    {
        printf("Enter the number of elements: ");
        scanf("%d", &n);
        array = (int *)malloc(n * sizeof(int));

        printf("Enter %d numbers:\n", n);
        for (int i = 0; i < n; i++)
            scanf("%d", &array[i]);
    }

    // Broadcast array size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate how many elements each process will handle
    int local_n = n / size;
    int rem = n % size;

    // Distribute remainder to first few processes
    if (rank < rem)
        local_n++;

    int *local_array = (int *)malloc(local_n * sizeof(int));

    // Scatter elements manually
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = n / size + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(array, sendcounts, displs, MPI_INT,
                 local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local sum
    int local_sum = 0;
    for (int i = 0; i < local_n; i++)
        local_sum += local_array[i];

    // Reduce all local sums to root
    int total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root computes average
    if (rank == 0)
    {
        double average = (double)total_sum / n;
        printf("Average of the array = %.2f\n", average);
    }

    // Free memory
    if (rank == 0)
    {
        free(array);
        free(sendcounts);
        free(displs);
    }
    free(local_array);

    MPI_Finalize();
    return 0;
}
