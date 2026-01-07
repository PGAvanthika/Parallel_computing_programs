#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main(int argc, char *argv[])
{
    int rank, size, n;
    int *marks = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Enforce exactly 3 processes
    if (size != 3)
    {
        if (rank == 0)
            printf("Run with exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // ---------------- ROOT INPUT ----------------
    if (rank == 0)
    {
        printf("Enter number of students (minimum 15): ");
        scanf("%d", &n);

        if (n < 15)
            MPI_Abort(MPI_COMM_WORLD, 1);

        marks = malloc(n * sizeof(int));
        printf("Enter student marks:\n");
        for (int i = 0; i < n; i++)
            scanf("%d", &marks[i]);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    int *local = malloc(local_n * sizeof(int));

    // ---------------- SCATTER ----------------
    MPI_Scatter(marks, local_n, MPI_INT,
                local, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    printf("\nP%d received: ", rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local[i]);
    printf("\n");

    // ---------------- LOCAL SORT ----------------
    qsort(local, local_n, sizeof(int), compare);

    printf("P%d local sort: ", rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local[i]);
    printf("\n");

    // ---------------- SAMPLE SELECTION ----------------
    int sample = local[local_n / 2];
    int samples[3];

    MPI_Gather(&sample, 1, MPI_INT,
               samples, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    // ---------------- SPLITTER SELECTION ----------------
    int splitters[2];
    if (rank == 0)
    {
        qsort(samples, 3, sizeof(int), compare);
        splitters[0] = samples[1];
        splitters[1] = samples[2];

        printf("\nSplitters chosen: %d %d\n",
               splitters[0], splitters[1]);
    }

    MPI_Bcast(splitters, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------------- BUCKET COUNTS ----------------
    int sendcounts[3] = {0, 0, 0};

    for (int i = 0; i < local_n; i++)
    {
        if (local[i] <= splitters[0])
            sendcounts[0]++;
        else if (local[i] <= splitters[1])
            sendcounts[1]++;
        else
            sendcounts[2]++;
    }

    // ---------------- BUCKET DATA ----------------
    int *sendbuf = malloc(local_n * sizeof(int));
    int index[3] = {0, sendcounts[0], sendcounts[0] + sendcounts[1]};
    int temp[3] = {0, 0, 0};

    for (int i = 0; i < local_n; i++)
    {
        int dest;
        if (local[i] <= splitters[0])
            dest = 0;
        else if (local[i] <= splitters[1])
            dest = 1;
        else
            dest = 2;

        sendbuf[index[dest] + temp[dest]] = local[i];
        temp[dest]++;
    }

    // ---------------- EXCHANGE COUNTS ----------------
    int recvcounts[3];
    MPI_Alltoall(sendcounts, 1, MPI_INT,
                 recvcounts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    int rdispls[3] = {0};
    int total_recv = recvcounts[0];

    for (int i = 1; i < 3; i++)
    {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        total_recv += recvcounts[i];
    }

    int *recvbuf = malloc(total_recv * sizeof(int));

    // ---------------- ALL-TO-ALL DATA EXCHANGE ----------------
    MPI_Alltoallv(sendbuf, sendcounts, index, MPI_INT,
                  recvbuf, recvcounts, rdispls, MPI_INT,
                  MPI_COMM_WORLD);

    // ---------------- FINAL LOCAL SORT ----------------
    qsort(recvbuf, total_recv, sizeof(int), compare);

    printf("P%d final bucket: ", rank);
    for (int i = 0; i < total_recv; i++)
        printf("%d ", recvbuf[i]);
    printf("\n");

    // ---------------- GATHER FINAL RESULT ----------------
    int *final = NULL;
    int *counts = NULL;
    int *displs = NULL;

    if (rank == 0)
    {
        final = malloc(n * sizeof(int));
        counts = malloc(3 * sizeof(int));
        displs = malloc(3 * sizeof(int));
    }

    MPI_Gather(&total_recv, 1, MPI_INT,
               counts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < 3; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
    }

    MPI_Gatherv(recvbuf, total_recv, MPI_INT,
                final, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // ---------------- FINAL OUTPUT ----------------
    if (rank == 0)
    {
        printf("\nFINAL SORTED MARKS:\n");
        for (int i = 0; i < n; i++)
            printf("%d ", final[i]);
        printf("\n");

        free(final);
        free(counts);
        free(displs);
        free(marks);
    }

    free(local);
    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
