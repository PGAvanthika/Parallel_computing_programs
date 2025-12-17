#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    // Write C code here
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("processor %d of size: %d", rank, size);

    MPI_Finalize();
    return 0;
}