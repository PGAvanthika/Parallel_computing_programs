#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


void bubbleSort(int *arr, int n)
{
    int temp;
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    int n;
    int *marks = NULL;
    int *sub_marks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (size != 3)
    {
        if (rank == 0)
            printf("Error: Run with exactly 3 processes.\n");
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


    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


    int counts[3], displs[3];
    int base = n / 3;
    int rem = n % 3;

    for (int i = 0; i < 3; i++)
        counts[i] = base + (i < rem ? 1 : 0);

    displs[0] = 0;
    for (int i = 1; i < 3; i++)
        displs[i] = displs[i - 1] + counts[i - 1];

    sub_marks = (int *)malloc(counts[rank] * sizeof(int));


    MPI_Scatterv(marks, counts, displs, MPI_INT,
                 sub_marks, counts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);


    printf("Process %d received: ", rank);
    for (int i = 0; i < counts[rank]; i++)
        printf("%d ", sub_marks[i]);
    printf("\n");


    bubbleSort(sub_marks, counts[rank]);

    printf("Process %d after local sort: ", rank);
    for (int i = 0; i < counts[rank]; i++)
        printf("%d ", sub_marks[i]);
    printf("\n");


    MPI_Gatherv(sub_marks, counts[rank], MPI_INT,
                marks, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        printf("\nGathered array at root (processor-wise):\n");
        for (int i = 0; i < n; i++)
            printf("%d ", marks[i]);
        printf("\n");


        bubbleSort(marks, n);

        printf("\nFinal globally sorted array:\n");
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
