#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int rowsA, colsA, rowsB, colsB;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("STEP 1: Enter matrix dimensions\n");
        printf("Enter rows and columns of Matrix A: ");
        scanf("%d %d", &rowsA, &colsA);

        printf("Enter rows and columns of Matrix B: ");
        scanf("%d %d", &rowsB, &colsB);

        if (colsA != rowsB)
        {
            printf("Matrix multiplication not possible\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (size < rowsA)
        {
            printf("Run with at least %d processes\n", rowsA);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int A[rowsA][colsA];
    int B[rowsB][colsB];
    int C[rowsA][colsB];
    int localA[colsA];
    int localC[colsB];

    if (rank == 0)
    {
        printf("\nSTEP 2: Enter elements of Matrix A\n");
        for (int i = 0; i < rowsA; i++)
            for (int j = 0; j < colsA; j++)
                scanf("%d", &A[i][j]);

        printf("\nSTEP 2: Enter elements of Matrix B\n");
        for (int i = 0; i < rowsB; i++)
            for (int j = 0; j < colsB; j++)
                scanf("%d", &B[i][j]);

        printf("\nMatrix A:\n");
        for (int i = 0; i < rowsA; i++)
        {
            for (int j = 0; j < colsA; j++)
                printf("%d ", A[i][j]);
            printf("\n");
        }

        printf("\nMatrix B:\n");
        for (int i = 0; i < rowsB; i++)
        {
            for (int j = 0; j < colsB; j++)
                printf("%d ", B[i][j]);
            printf("\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("\nSTEP 3: Scattering rows of Matrix A\n");

    MPI_Scatter(A, colsA, MPI_INT, localA, colsA, MPI_INT, 0, MPI_COMM_WORLD);

    for (int p = 0; p < rowsA; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p)
        {
            printf("Process %d received row %d of A: ", rank, rank);
            for (int i = 0; i < colsA; i++)
                printf("%d ", localA[i]);
            printf("\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("\nSTEP 4: Broadcasting Matrix B\n");

    MPI_Bcast(B, rowsB * colsB, MPI_INT, 0, MPI_COMM_WORLD);

    for (int p = 0; p < rowsA; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p)
            printf("Process %d received Matrix B\n", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int p = 0; p < rowsA; p++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p)
        {
            printf("\nSTEP 5: Process %d computing row %d of Matrix C\n", rank, rank);
            for (int j = 0; j < colsB; j++)
            {
                localC[j] = 0;
                printf("C[%d][%d] = ", rank, j);
                for (int k = 0; k < colsA; k++)
                {
                    localC[j] += localA[k] * B[k][j];
                    printf("(%d*%d)", localA[k], B[k][j]);
                    if (k < colsA - 1)
                        printf(" + ");
                }
                printf(" = %d\n", localC[j]);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("\nSTEP 6: Gathering result matrix\n");

    MPI_Gather(localC, colsB, MPI_INT, C, colsB, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\nSTEP 7: Final Result Matrix C = A × B\n");
        for (int i = 0; i < rowsA; i++)
        {
            for (int j = 0; j < colsB; j++)
                printf("%d ", C[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}