#include <iostream>
#include "mpi.h"
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
using namespace std;
/*
module purge
module load gcc/4.9.0
module load cmake/3.9.1
module load openmpi

mpirun -np 4 ./heat1D 10 10 8 5

5.48828, 2.26562, 0.654297, 0.126953, 0.126953, 0.654297, 2.26562, 5.48828

*/

// global vars
double h = 2;
double k = 1;
double r = k / (h * h);

int main(int argc, char** argv) {
    // terminal input parameters
    // T1temp T2temp NumGridPoints NumTimesteps
    double T1 = strtod(argv[1], NULL);
    double T2 = strtod(argv[2], NULL);
    int NumGridPoints = strtol(argv[3], NULL, 10) + 2;
    int NumTimesteps = strtol(argv[4], NULL, 10);
    int numtasks, rank, rc, sizearr;
    double* arr;
    double* temp;
    double buf, buf0, left, right;
    MPI_Status status;

    // Initialize the MPI environment
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error occurred initializing MPI program. Abort!\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Get the number of processes
    // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf ("\nNumber of tasks = %d My rank = %d\n", numtasks, rank);

    // initializing array (arr) for each process
    if (NumGridPoints >= numtasks) {
        if (NumGridPoints % numtasks == 0) {
            sizearr = NumGridPoints / numtasks;
            arr = new double[sizearr];
            temp = new double[sizearr];
            for(int i = 0; i < (sizearr); i++) {
                temp[i] = 0;
                arr[i] = 0;
            }
            if (rank == 0) {
                arr[0] = T1;
                temp[0] = T1;
            }
            if (rank == numtasks - 1) {
                temp[sizearr - 1] = T2;
                arr[sizearr - 1] = T2;
            }

            // cout << "case perfectly devisible, rank = " << rank << endl;
            // cout << "size of array = " << NumGridPoints / numtasks << endl << endl;
        } else {
            if (rank != numtasks - 1) {
                sizearr = NumGridPoints / numtasks;
                arr = new double[sizearr];
                temp = new double[sizearr];
                for(int i = 0; i < (sizearr); i++) {
                    temp[i] = 0;
                    arr[i] = 0;
                }
                // cout << "case rank = " << rank << endl;
                // cout << "size of array = " << NumGridPoints / numtasks << endl << endl;
            } else {
                sizearr = NumGridPoints / numtasks + NumGridPoints % numtasks;
                arr = new double[sizearr];
                temp = new double[sizearr];
                for(int i = 0; i < sizearr; i++) {
                    temp[i] = 0;
                    arr[i] = 0;
                }
                // cout << "last rank = " << rank << endl;
                // cout << "size of array = " << sizearr << endl << endl;
                if (rank == 0) {
                    arr[0] = T1;
                    temp[0] = T1;
                }
                if (rank == numtasks - 1) {
                    temp[sizearr - 1] = T2;
                    arr[sizearr - 1] = T2;
                }
            }
        }
    } else {
        sizearr = 1;
        arr = new double[sizearr];
        temp = new double [sizearr];
        numtasks = NumGridPoints;
        for(int i = 0; i < (sizearr); i++) {
            temp[i] = 0;
            arr[i] = 0;
        }
        if (rank == 0) {
            arr[0] = T1;
            temp[0] = T1;
        }
        if (rank == numtasks - 1) {
            temp[sizearr - 1] = T2;
            arr[sizearr - 1] = T2;
        }
        if (rank >= NumGridPoints) {
            cout << "I AM RANK " << rank << " AND I AM KILLING MYSELF***************" << endl;
            MPI_Finalize();
            return 0;
        }
    }




    // special case where there is only one rank
    if (numtasks == 1) {
        // actual calculation
        for (int n = 0; n < (NumTimesteps); n++) {
            for(int i = 0; i < (NumGridPoints); i++)
                temp[i] = arr[i];

            for (int j = 1; j < (NumGridPoints - 1); j++) {
                temp[j] = (1 - 2 * r) * arr[j] + r * arr[j - 1] + r * arr[j + 1];
                // cout << arr[j - 1] << " " << arr[j] << " " << arr[j + 1] << endl;
            }
            for(int i = 0; i < (NumGridPoints); i++)
                arr[i] = temp[i];
        }

        // convert to csv file
        ofstream myFile;
        myFile.open("heat1Doutput.csv");
        for(int i = 1; i < (NumGridPoints - 2); i++)
            myFile << arr[i] << ", ";
        myFile << arr[NumGridPoints - 2];
        MPI_Finalize();
        return 0;
    }


    // if numtasks > 1, do this
    for (int round = 0; round < NumTimesteps; round++) {
        // cout << endl << endl << "WE ARE AT ROUND !!!!!!!!!!!!!!!  " << round << endl;
        // odd ranks receive first, and then send, avoiding deadlock
        if (rank % 2 == 0) {
            cout << "rank"<< rank<< " even: round" << round << endl;
            // debugging
            // cout << "Rank " << rank
            //      << " sending to rank" << rank + 1
            //      << " round " << round << endl;
            // sending first
            if (rank != 0) {
                // set buf equal to first element in arr
                buf = arr[0];
                // cout << "trying to send .................." << endl;
                rc = MPI_Send(&buf, 1, MPI_DOUBLE, rank - 1,
                              0, MPI_COMM_WORLD);
                cout << "rank" << rank << " sending data = " << buf << " to rank" << rank - 1 << endl;

                // cout << "trying to receive .................." << endl;
                rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, rank - 1,
                              3, MPI_COMM_WORLD, &status);
                cout << "rank" << rank << " receiving data = " << buf0 << " from rank" << rank - 1 << endl;

                left = buf0;
            } else {
                left = T1;
            }

            if (rank != numtasks - 1) {
                // set buf equal to last element in arr
                buf = arr[sizearr - 1];
                // cout << "trying to send .................." << buf << endl;
                rc = MPI_Send(&buf, 1, MPI_DOUBLE, rank + 1,
                              1, MPI_COMM_WORLD);
                cout << "rank" << rank << " sending data = " << buf << " to rank" << rank + 1 << endl;


                // cout << "trying to receive .................." << endl;
                rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, rank + 1,
                              2, MPI_COMM_WORLD, &status);
                cout << "rank" << rank << " receiving data = " << buf0 << " from rank" << rank + 1 << endl;

                right = buf0;
            } else {
                right = T2;
            }
        }
        else {
            cout << "rank"<< rank<< " odd: round" << round << endl;
            // receiving first

            // cout << "trying to receive .................." << endl;
            rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, rank - 1,
                          1, MPI_COMM_WORLD, &status);
            cout << "rank" << rank << " receiving data = " << buf0 << " from rank" << rank - 1 << endl;
            // send rank - 1 no matter what

            buf = arr[0];
            // cout << "trying to send .................." << endl;
            rc = MPI_Send(&buf, 1, MPI_DOUBLE, rank - 1,
                          2, MPI_COMM_WORLD);
            cout << "rank" << rank << " sending data = " << buf << " to rank" << rank - 1 << endl;
            left = buf0;

            if (rank != numtasks - 1) {
                // cout << "trying to receive .................." << endl;

                rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, rank + 1,
                              0, MPI_COMM_WORLD, &status);
                cout << "rank" << rank << " receiving data = " << buf0 << " from rank" << rank + 1 << endl;
                // sent to rank + 1

                buf = arr[sizearr - 1];
                // cout << "trying to send .................." << endl;
                rc = MPI_Send(&buf, 1, MPI_DOUBLE, rank + 1,
                              3, MPI_COMM_WORLD);
                cout << "rank" << rank << " sending data = " << buf << " to rank" << rank + 1 << endl;
                right = buf0;
            } else {
                right = T2;
            }

            // Now send to next rank (0 if we are last rank)

        }

        for (int i = 0; i < sizearr; i++) {
            temp[i] = arr[i];
        }
        if (rank == 0) {
            for (int i = 1; i < sizearr; i++) {
                if (sizearr == 2) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * right;
                } else if (i == 1) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * arr[i + 1];
                } else if (i == sizearr - 1) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * right;
                } else {
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * arr[i + 1];
                }
            }
        } else if (rank == numtasks - 1) {
            for (int i = 0; i < sizearr - 1; i++) {
                if (sizearr == 1) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * right;
                } else if (i == sizearr - 2) {
                    // cout << "RANK 3 SHOULD BE PRINTING THIS..........." << endl;
                    // cout << "arr[i] = " << arr[i] << "arr[i - 1] = " << arr[i-1] << "right = " << right << endl;
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * right;
                } else if (i == 0) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * arr[i + 1];
                } else {
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * arr[i + 1];
                }
            }
        } else {
            for (int i = 0; i < sizearr; i++) {
                if (sizearr == 1) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * right;
                } else if (i == 0) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * left + r * arr[i + 1];
                } else if (i == sizearr - 1) {
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * right;
                } else {
                    temp[i] = (1 - 2 * r) * arr[i] + r * arr[i - 1] + r * arr[i + 1];
                }
            }
        }

        cout << "FOR RANK " << rank << ":  ";
        for(int i = 0; i < sizearr; i++) {
            arr[i] = temp[i];
            cout << arr[i] << ", ";
        }
        cout << "@ round" << round << endl << endl;

    }


    // sending all data one by one back to rank 0
    if (rank != 0) {
        for (int i = 0; i < sizearr; i++) {
            buf = arr[i];
            cout << "ERROR1: sending from rank = " << rank << "data = " << buf << endl;
            rc = MPI_Send(&buf, 1, MPI_DOUBLE, 0,
                                  0, MPI_COMM_WORLD);
        }
        cout << "ERROR2: sent successful from rank" << rank << endl;
    } else {

        int length = sizearr;
        for (int i = 1; i < numtasks; i++) {

            cout << "ERROR3: preparing to receive from rank" << i << endl;
            if (i != numtasks - 1) {
                for (int j = 0; j < sizearr; j++) {
                    rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, i,
                                      0 ,MPI_COMM_WORLD, &status);
                    cout << "ERROR4: receive successful from rank" << i << ", data at position " << j << ", data = " << buf0 << endl;
                    arr[length] = buf0;
                    length++;
                    cout << "j = " << j << ", sizearr = " << sizearr << endl;
                    cout << "length = " << length << endl;
                }
            } else {
                for (int j = 0; j < NumGridPoints / numtasks + NumGridPoints % numtasks; j++) {
                    rc = MPI_Recv(&buf0, 1, MPI_DOUBLE, i,
                                      0 ,MPI_COMM_WORLD, &status);
                    cout << "ERROR5: receive successful from last rank" << i << ", data at position " << j << ", data = " << buf0 << endl;
                    arr[length] = buf0;
                    length++;
                    cout << "j = " << j << ", sizelast = " << NumGridPoints / numtasks + NumGridPoints % numtasks << endl;
                }
            }
        }

        cout << "we are at rank 0, and the value in arr are: " << endl;
        // saving to csv
        ofstream myFile;
        myFile.open("heat1Doutput.csv");
        for (int i = 1; i < length - 2; i++) {
            myFile << arr[i] << ", ";
            cout << arr[i] << endl;
        }
        myFile << arr[length - 2];
        cout << arr[length - 2] << endl;
        if(myFile != NULL) {
            myFile.close();
        }


        cout << "ERROR6: all sends successful" << endl;

    }

    cout << "Rank " << rank << " exiting normally" << endl;

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}