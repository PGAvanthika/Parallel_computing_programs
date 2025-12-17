#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int msg;
    int ping_pongs = 10; // Number of ping-pong exchanges

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (size != 2)
    {
        if (rank == 0)
        {
            printf("This program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        msg = 10;
        for (int i = 0; i < ping_pongs; i++)
        {
            printf("Process 0 sending message %d to Process 1\n", msg);
            MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 0 received message %d from Process 1\n", msg);
            msg++;
        }
    }
    else if (rank == 1)
    {
        for (int i = 0; i < ping_pongs; i++)
        {
            MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process 1 received message %d from Process 0\n", msg);
            msg++;
            MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Process 1 sent message %d to Process 0\n", msg);
        }
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}