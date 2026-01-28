#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Compare function for qsort */
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/* Merge 3 sorted arrays */
void merge3(int *a, int na, int *b, int nb, int *c, int nc, int *out)
{
    int i = 0, j = 0, k = 0, p = 0;
    while (i < na || j < nb || k < nc)
    {
        int min = 1000000000;
        if (i < na && a[i] < min)
            min = a[i];
        if (j < nb && b[j] < min)
            min = b[i] < min ? b[j] : min;
        if (k < nc && c[k] < min)
            min = c[k] < min ? c[k] : min;

        if (i < na && a[i] == min)
            out[p++] = a[i++];
        else if (j < nb && b[j] == min)
            out[p++] = b[j++];
        else
            out[p++] = c[k++];
    }
}

int main(int argc, char *argv[])
{
    int rank, size, n;
    int *marks = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3)
    {
        if (rank == 0)
            printf("Run with exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }
    

    /* ---------- INPUT AT ROOT ---------- */
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

    /* Broadcast number of elements */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Divide work evenly */
    int local_n = n / size;
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatter(marks, local_n, MPI_INT,
                local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    printf("\nP%d received: ", rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local[i]);
    printf("\n");

    /* Local sort */
    qsort(local, local_n, sizeof(int), compare);
    printf("P%d after local sort: ", rank);
    for (int i = 0; i < local_n; i++)
        printf("%d ", local[i]);
    printf("\n");

    /* Sample selection (middle element) */
    int sample = local[local_n / 2];
    int samples[3];

    MPI_Gather(&sample, 1, MPI_INT, samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Splitter selection at root */
    int splitters[2];
    if (rank == 0)
    {
        qsort(samples, 3, sizeof(int), compare);
        splitters[0] = samples[1];
        splitters[1] = samples[2];
        printf("\nSplitters selected: %d %d\n", splitters[0], splitters[1]);
    }

    MPI_Bcast(splitters, 2, MPI_INT, 0, MPI_COMM_WORLD);

    /* Create buckets based on splitters */
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

    int *sendbuf = malloc(local_n * sizeof(int));
    int sdispls[3] = {0, sendcounts[0], sendcounts[0] + sendcounts[1]};
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
        sendbuf[sdispls[dest] + temp[dest]] = local[i];
        temp[dest]++;
    }

    /* Print unsorted buckets */
    printf("P%d buckets (unsorted within process):\n", rank);
    for (int b = 0; b < 3; b++)
    {
        printf("Bucket%d: ", b);
        for (int i = sdispls[b]; i < sdispls[b] + sendcounts[b]; i++)
            printf("%d ", sendbuf[i]);
        printf("\n");
    }

    /* Exchange bucket counts */
    int recvcounts[3];
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int rdispls[3] = {0};
    int total_recv = recvcounts[0];
    for (int i = 1; i < 3; i++)
    {
        rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
        total_recv += recvcounts[i];
    }

    int *recvbuf = malloc(total_recv * sizeof(int));

    /* All-to-all data exchange */
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT,
                  recvbuf, recvcounts, rdispls, MPI_INT,
                  MPI_COMM_WORLD);

    /* Print received bucket before final sort */
    printf("P%d received bucket (unsorted): ", rank);
    for (int i = 0; i < total_recv; i++)
        printf("%d ", recvbuf[i]);
    printf("\n");

    /* Final local sort */
    qsort(recvbuf, total_recv, sizeof(int), compare);

    /* Print final sorted bucket */
    printf("P%d final sorted bucket: ", rank);
    for (int i = 0; i < total_recv; i++)
        printf("%d ", recvbuf[i]);
    printf("\n");

    /* Gather sorted buckets at root */
    int *counts = NULL, *displs = NULL, *final = NULL;
    if (rank == 0)
    {
        counts = malloc(3 * sizeof(int));
        displs = malloc(3 * sizeof(int));
    }

    MPI_Gather(&total_recv, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < 3; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        final = malloc(n * sizeof(int));
    }

    MPI_Gatherv(recvbuf, total_recv, MPI_INT,
                final, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Merge 3 buckets at root */
    if (rank == 0)
    {
        int *merged = malloc(n * sizeof(int));
        merge3(final + displs[0], counts[0],
               final + displs[1], counts[1],
               final + displs[2], counts[2],
               merged);

        printf("\nFINAL SORTED MARKS (MERGED AT ROOT):\n");
        for (int i = 0; i < n; i++)
            printf("%d ", merged[i]);
        printf("\n");

        free(merged);
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
