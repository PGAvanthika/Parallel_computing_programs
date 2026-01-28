#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
void merge(int *a, int n1, int *b, int n2, int *result)
{
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2)
        result[k++] = (a[i] < b[j]) ? a[i++] : b[j++];

    while (i < n1)
        result[k++] = a[i++];

    while (j < n2)
        result[k++] = b[j++];
}
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}
int main(int argc, char *argv[])
{
    int rank, size, n;
    int *data = NULL;
    int *localData = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
    {
        printf("Enter number of elements (minimum 15): ");
        fflush(stdout);
        scanf("%d", &n);
        if (n < 15)
        {
            printf("Please enter at least 15 number of elements\n");
            fflush(stdout);
            MPI_Finalize();
            return 0;
        }
        data = (int *)malloc(n * sizeof(int));
        printf("Enter %d numbers:\n", n);
        fflush(stdout);
        for (int i = 0; i < n; i++)
            scanf("%d", &data[i]);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int base = n / size;
    int rem = n % size;
    int local_n = base + (rank < rem ? 1 : 0);
    localData = (int *)malloc(local_n * sizeof(int));
    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    MPI_Scatterv(data, sendcounts, displs, MPI_INT,
                 localData, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);
    qsort(localData, local_n, sizeof(int), compare);
    for (int p = 0; p < size; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p)
        {
            printf("Process %d sorted subarray: ", rank);
            for (int i = 0; i < local_n; i++)
                printf("%d ", localData[i]);
            printf("\n");
            fflush(stdout);
        }
    }
    MPI_Gatherv(localData, local_n, MPI_INT,
                data, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        int *temp = (int *)malloc(n * sizeof(int));
        int current_size = sendcounts[0];

        for (int i = 1; i < size; i++)
        {
            merge(data, current_size,
                  data + displs[i], sendcounts[i],
                  temp);

            for (int j = 0; j < current_size + sendcounts[i]; j++)
                data[j] = temp[j];

            current_size += sendcounts[i];
        }

        printf("\nFinal Sorted Array:\n");
        for (int i = 0; i < n; i++)
            printf("%d ", data[i]);
        printf("\n");
        free(temp);
        free(data);
        free(sendcounts);
        free(displs);
    }
    free(localData);
    MPI_Finalize();
    return 0;
}