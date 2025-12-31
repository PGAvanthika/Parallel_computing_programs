#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Comparator function for qsort */
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

int main(int argc, char *argv[])
{
    int rank, size;
    int n;
    int *marks = NULL; // Full array (root)
    int *sub_marks;    // Local array

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Enforce exactly 3 processes
    if (size != 3)
    {
        if (rank == 0)
            printf("Error: This program must be run with exactly 3 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0)
    {
        printf("Enter number of students: ");
        scanf("%d", &n);

        marks = (int *)malloc(n * sizeof(int));
        printf("Enter student marks:\n");
        for (int i = 0; i < n; i++)
            scanf("%d", &marks[i]);
    }

    // Broadcast n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine counts and displacements for Scatterv
    int counts[3], displs[3];
    int base = n / 3;
    int rem = n % 3;
    for (int i = 0; i < 3; i++)
        counts[i] = base + (i < rem ? 1 : 0);

    displs[0] = 0;
    for (int i = 1; i < 3; i++)
        displs[i] = displs[i - 1] + counts[i - 1];

    sub_marks = (int *)malloc(counts[rank] * sizeof(int));

    // Scatter data
    MPI_Scatterv(marks, counts, displs, MPI_INT,
                 sub_marks, counts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    // Show what each process received
    printf("Process %d received: ", rank);
    for (int i = 0; i < counts[rank]; i++)
        printf("%d ", sub_marks[i]);
    printf("\n");

    // Local sort
    qsort(sub_marks, counts[rank], sizeof(int), compare);

    // Show local sorted array
    printf("Process %d after local sort: ", rank);
    for (int i = 0; i < counts[rank]; i++)
        printf("%d ", sub_marks[i]);
    printf("\n");

    // Gather sorted subarrays
    MPI_Gatherv(sub_marks, counts[rank], MPI_INT,
                marks, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Final sort at root
    if (rank == 0)
    {
        qsort(marks, n, sizeof(int), compare);
        printf("\nFinal sorted array at root:\n");
        for (int i = 0; i < n; i++)
            printf("%d ", marks[i]);
        printf("\n");
    }

    free(sub_marks);
    if (rank == 0)
        free(marks);

    MPI_Finalize();
    return 0;
}
