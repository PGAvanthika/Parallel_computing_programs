#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Status status;
    int msg;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;        // next process
    int prev = (rank - 1 + size) % size; // previous process

    if (rank == 0)
    {
        msg = 100; // initial message
        printf("Process %d sending %d to %d\n", rank, msg, next);

        MPI_Send(&msg, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&msg, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);

        printf("Process %d received %d from %d\n", rank, msg, prev);
    }
    else
    {
        MPI_Recv(&msg, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        printf("Process %d received %d from %d\n", rank, msg, prev);


        MPI_Send(&msg, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        printf("Process %d sent %d to %d\n", rank, msg, next);
    }

    MPI_Finalize();
    return 0;
}
