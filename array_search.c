#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int n, key;
    int array[100], local_array[100];
    int sendcounts[3], displs[3];

    int local_indices[100];
    int local_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 3)
    {
        if (rank == 0)
            printf("Error: Processes must be <= 3\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        printf("Enter number of elements:\n");
        scanf("%d", &n);

        printf("Enter array elements:\n");
        for (int i = 0; i < n; i++)
            scanf("%d", &array[i]);

        printf("Enter element to search:\n");
        scanf("%d", &key);

        int base = n / size, rem = n % size, offset = 0;
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Bcast(&key, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = sendcounts[rank];

    MPI_Scatterv(array, sendcounts, displs, MPI_INT,
                 local_array, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    printf("Process %d received elements: ", rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local_array[i]);
    printf("\n");

    /* Local search for duplicates */
    for (int i = 0; i < local_n; i++)
    {
        if (local_array[i] == key)
        {
            local_indices[local_count++] = displs[rank] + i;
            printf("Process %d found element at index %d\n",
                   rank, displs[rank] + i);
        }
    }

    int recv_counts[3];
    MPI_Gather(&local_count, 1, MPI_INT,
               recv_counts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    int total_matches = 0;
    int recv_displs[3];

    if (rank == 0)
    {
        recv_displs[0] = 0;
        total_matches = recv_counts[0];
        for (int i = 1; i < size; i++)
        {
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
            total_matches += recv_counts[i];
        }
    }

    int all_indices[100];

    MPI_Gatherv(local_indices, local_count, MPI_INT,
                all_indices, recv_counts, recv_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (total_matches > 0)
        {
            printf("Final Result: Element found at indices: ");
            for (int i = 0; i < total_matches; i++)
                printf("%d ", all_indices[i]);
            printf("\n");
        }
        else
        {
            printf("Final Result: Element not found\n");
        }
    }

    MPI_Finalize();
    return 0;
}
