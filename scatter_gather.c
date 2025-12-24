#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int n;
    int *array = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure exactly 3 processes
    if (size != 3)
    {
        if (rank == 0)
            printf("This program must be run with exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // Root takes input
    if (rank == 0)
    {
        printf("Enter array size: ");
        scanf("%d", &n);

        array = (int *)malloc(n * sizeof(int));

        printf("Enter %d elements:\n", n);
        for (int i = 0; i < n; i++)
            scanf("%d", &array[i]);

        // Allocate send counts and displacements
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int base = n / size;
        int rem = n % size;

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = base;
            if (i < rem)
                sendcounts[i]++;

            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    // Broadcast sendcounts to all processes
    int local_n;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_array = (int *)malloc(local_n * sizeof(int));

    // Scatter uneven data
    MPI_Scatterv(array, sendcounts, displs, MPI_INT,
                 local_array, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Compute local sum
    int local_sum = 0;
    for (int i = 0; i < local_n; i++)
        local_sum += local_array[i];

    printf("Process %d local sum = %d\n", rank, local_sum);

    // Gather local sums
    int *partial_sums = NULL;
    if (rank == 0)
        partial_sums = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_sum, 1, MPI_INT,
               partial_sums, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // Final sum
    if (rank == 0)
    {
        int total_sum = 0;
        for (int i = 0; i < size; i++)
            total_sum += partial_sums[i];

        printf("Final sum of array = %d\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}
