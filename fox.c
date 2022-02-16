#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

#define DEBUG 0
#define DT_MATRIX unsigned long
#define DT_MATRIX_MPI MPI_UNSIGNED_LONG
#define MAX_MATRIX_VALUE INT_MAX

// Function to return the minimun of two numbers
DT_MATRIX minimum (DT_MATRIX a, DT_MATRIX b);

// Function to copy the result array to the inputs arrays
void copyArrays(DT_MATRIX * pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size);

// Function to correct the result from int_max to 0
void correctResult(DT_MATRIX * pCMatrix, int Size);

// Function for receiving the input file of the matrix
bool getInputFileName(int argc, char* argv[], char **input);

// Function to read the input file, allocate memory for the blocks and matrices and init the data
void ProcessInitialization (DT_MATRIX** pAMatrix, DT_MATRIX** pBMatrix, DT_MATRIX** pCMatrix, DT_MATRIX** pAblock, DT_MATRIX** pBblock, DT_MATRIX** pCblock,DT_MATRIX** pMatrixAblock, int *Size, int *BlockSize, char* input );

// Function for printing a matrix
void PrintMatrix(DT_MATRIX *pMatrix, int Size);

// Function to clean the environment and memory allocated
void ProcessTermination (DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, DT_MATRIX* pAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock,DT_MATRIX* pMatrixAblock);

// Function to do a typical serial calculation on a matrix O(n^3)
void SerialResultCalculation(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size);

// Function for the special matrix multiplication
void SpecialMatrixMultiply(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size);

// Function to initialize the matrix with the data inside the input file
void DataInitialization(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size, char * input);

// Function for creating the two-dimensional grid communicator and communicators for each row and each column of the grid
void CreateGridCommunicators();

// Function for checkerboard matrix decomposition
void CheckerboardMatrixScatter(DT_MATRIX* pMatrix, DT_MATRIX* pMatrixBlock, int Size, int BlockSize);

// Function for data distribution among the processes
void DataDistribution(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, DT_MATRIX* pMatrixAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock, int Size, int BlockSize);

// Test printing of the matrix block
void TestBlocks (DT_MATRIX* pBlock, int BlockSize, char str[]);

// Fox's algorithm main method
void ParallelResultCalculation(DT_MATRIX* pAblock, DT_MATRIX* pMatrixAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock, int BlockSize);

// Broadcasting blocks of the matrix A to process grid rows
void ABlockCommunication (int iter, DT_MATRIX *pAblock, DT_MATRIX* pMatrixAblock, int BlockSize);

// Function for cyclic shifting the blocks of the matrix B
void BblockCommunication (DT_MATRIX *pBblock, int BlockSize, MPI_Comm ColumnComm);

// Function for block multiplication
void BlockMultiplication(DT_MATRIX* pAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock, int Size);

// Function for gathering the result matrix
void ResultCollection (DT_MATRIX* pCMatrix, DT_MATRIX* pCblock, int Size, int BlockSize);


int ProcNum = 0;                // Number of available processes
int ProcRank = 0;               // Rank of current process

// Grid settings and comunicators
int GridSize;                   // Size of virtual processor grid
MPI_Comm GridComm;              // Grid communicaror
int GridCoords[2];              // Coordinates of current processor in grid

// Columns and rows of the comunicator
MPI_Comm ColComm;               // Column communicator
MPI_Comm RowComm;               // Row comunicator

int main(int argc, char *argv[]) {
    
    // General vars
    DT_MATRIX* pAMatrix;        // First argument of matrix multiplication
    DT_MATRIX* pBMatrix;        // Second argument of matrix multiplication
    DT_MATRIX* pCMatrix;        // Result matrix
    int Size;                   // Size of matrices
    double Start, Finish, Duration; // Profiling vars
    
    // Blocks of matrices of each block
    int BlockSize;             // Sizes of matrix blocks
    DT_MATRIX *pMatrixAblock;  // Initial block of matrix A
    DT_MATRIX *pAblock;        // Current block of matrix A
    DT_MATRIX *pBblock;        // Current block of matrix B
    DT_MATRIX *pCblock;        // Block of result matrix C
    
    char* input;               // The input file name
    int iteration = 1;         // The number of iterations until Df = DgxDg
    
    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    // Validating the grid size to be the perfect sqrt of the number of processes
    GridSize = sqrt((double)ProcNum);
    if (ProcNum != GridSize*GridSize) {
        if (ProcRank == 0) {
            printf ("ERROR: Invalid configuration! \n");
        }
    }
    else
    {
        // Reading the argument to get the input file
        if (ProcRank == 0)
        {
            bool bHasFilename = getInputFileName(argc, argv, &input);
            /*if (DEBUG)
             printf("Got filename = %s\n", input);
             */
            if (!bHasFilename){
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Memory allocation and initialization of matrix elements
        ProcessInitialization(&pAMatrix, &pBMatrix, &pCMatrix, &pAblock, &pBblock, &pCblock, &pMatrixAblock, &Size, &BlockSize, input);
        
        // Createing grid communicators
        CreateGridCommunicators();
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        /*if (DEBUG && ProcRank == 0){
         printf("Initial matrix A \n");
         PrintMatrix(pAMatrix, Size);
         printf("Initial matrix B \n");
         PrintMatrix(pBMatrix, Size);
         }*/
        
        if (DEBUG)
            Start = MPI_Wtime();
                
        do {
            // Data distribution among the processes
            DataDistribution(pAMatrix, pBMatrix, pCMatrix, pMatrixAblock, pBblock, pCblock, Size, BlockSize);
            
            /*if (DEBUG){
             TestBlocks(pMatrixAblock, BlockSize, "Initial blocks of matrix A");
             TestBlocks(pBblock, BlockSize, "Initial blocks of matrix B");
             }*/
            
            ParallelResultCalculation(pAblock, pMatrixAblock, pBblock, pCblock, BlockSize);
            /*if (DEBUG) {
             TestBlocks(pCblock, BlockSize, "Result blocks");
             }*/
            
            // Gathering the result matrix
            ResultCollection(pCMatrix, pCblock, Size, BlockSize);
            
            if (ProcRank == 0){
                memcpy(pAMatrix, pCMatrix, sizeof(DT_MATRIX) * Size * Size);
                memcpy(pAMatrix, pCMatrix, sizeof(DT_MATRIX) * Size * Size);

                //copyArrays(pAMatrix, pBMatrix, pCMatrix, Size);
                iteration = iteration * 2;
            }
            MPI_Bcast(&iteration, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
        }
        while (iteration < Size);
        
        if (DEBUG) {
            Finish = MPI_Wtime();
            Duration = Finish-Start;
            if (ProcRank == 0) {
                printf("Time of execution = %f\n", Duration);
            }
        }
        if (ProcRank == 0) {
            correctResult(pCMatrix, Size);
            PrintMatrix(pCMatrix, Size);
        }
        
        // Process termination
        ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock, pCblock, pMatrixAblock);
    }
    
    MPI_Finalize();
    return 0;
}

// Function to correct the result from int_max to 0
void correctResult(DT_MATRIX * pCMatrix, int Size) {
    int i, j;  // Loop variables
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++) {
            if(pCMatrix[i*Size+j] == MAX_MATRIX_VALUE)
                pCMatrix[i*Size+j] = 0;
        }
    }
}

DT_MATRIX minimum (DT_MATRIX a, DT_MATRIX b) {
    if (a < b)
        return a;
    return b;
}


bool getInputFileName(int argc, char* argv[], char **input) {
    if (argc < 2){
        printf("ERROR: You must run the program as 'mpirun -np {numprocess} program {filename}");
        return false;
    }
    
    *input = argv[1];
    return true;
}


// Test printing of the matrix block
void TestBlocks (DT_MATRIX* pBlock, int BlockSize, char str[]) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("%s \n", str);
    }
    for (int i=0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf ("ProcRank = %d \n", ProcRank);
            PrintMatrix(pBlock, BlockSize);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Function for parallel execution of the Fox method
void ParallelResultCalculation(DT_MATRIX* pAblock, DT_MATRIX* pMatrixAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock, int BlockSize) {
    for (int iter = 0; iter < GridSize; iter ++) {
        
        // Sending blocks of matrix A to the process grid rows
        ABlockCommunication(iter, pAblock, pMatrixAblock, BlockSize);
        /*if (DEBUG){
         if (ProcRank == 0)
         printf("Iteration number %d \n", iter);
         TestBlocks(pAblock, BlockSize, "Block of A matrix");
         }*/
        
        // Block multiplication
        BlockMultiplication( pAblock, pBblock, pCblock, BlockSize );
        
        // Cyclic shift of blocks of matrix B in process grid columns
        BblockCommunication ( pBblock, BlockSize, ColComm );
        /*if (DEBUG) {
         if (ProcRank == 0)
         printf("Iteration number %d \n", iter);
         TestBlocks(pBblock, BlockSize, "Block of B matrix");
         }*/
        
        //TestBlocks(pCblock, BlockSize, "Block of C matrix");

    }
}

// Function for gathering the result matrix
void ResultCollection (DT_MATRIX* pCMatrix, DT_MATRIX* pCblock, int Size, int BlockSize) {
    DT_MATRIX * pResultRow = malloc(sizeof(DT_MATRIX) * Size * BlockSize);
    for (int i=0; i < BlockSize; i++) {
        MPI_Gather( &pCblock[i*BlockSize], BlockSize, DT_MATRIX_MPI, &pResultRow[i*Size], BlockSize, DT_MATRIX_MPI, 0, RowComm);
    }
    if (GridCoords[1] == 0) {
        MPI_Gather(pResultRow, BlockSize*Size, DT_MATRIX_MPI, pCMatrix, BlockSize*Size, DT_MATRIX_MPI, 0, ColComm);
    }
    free(pResultRow);
}

// Broadcasting blocks of the matrix A to process grid rows
void ABlockCommunication (int iter, DT_MATRIX *pAblock, DT_MATRIX* pMatrixAblock, int BlockSize) {  // Defining the leading process of the process grid row
    int Pivot = (GridCoords[0] + iter) % GridSize;
    // Copying the transmitted block in a separate memory buffer
    if (GridCoords[1] == Pivot) {
        for (int i = 0; i < BlockSize*BlockSize; i++)
        pAblock[i] = pMatrixAblock[i];
    }
    // Block broadcasting
    MPI_Bcast(pAblock, BlockSize * BlockSize, DT_MATRIX_MPI, Pivot, RowComm);
}

// Function for cyclic shifting the blocks of the matrix B
void BblockCommunication (DT_MATRIX *pBblock, int BlockSize, MPI_Comm ColumnComm) {
    MPI_Status Status;
    
    int NextProc = GridCoords[0] + 1;
    if ( GridCoords[0] == GridSize-1 )
        NextProc = 0;
    
    int PrevProc = GridCoords[0] - 1;
    if ( GridCoords[0] == 0 )
        PrevProc = GridSize-1;
    
    MPI_Sendrecv_replace( pBblock, BlockSize*BlockSize, DT_MATRIX_MPI, NextProc, 0, PrevProc, 0, ColumnComm, &Status);
}

// Function for block multiplication
void BlockMultiplication(DT_MATRIX* pAblock, DT_MATRIX* pBblock,DT_MATRIX* pCblock, int Size) {
    SpecialMatrixMultiply(pAblock, pBblock, pCblock, Size);
}

// Function for checkerboard matrix decomposition
void CheckerboardMatrixScatter(DT_MATRIX* pMatrix, DT_MATRIX* pMatrixBlock, int Size, int BlockSize)
{
    DT_MATRIX * pMatrixRow = malloc(sizeof(DT_MATRIX) * BlockSize * Size);
    if (GridCoords[1] == 0) {
        MPI_Scatter(pMatrix, BlockSize * Size, DT_MATRIX_MPI, pMatrixRow, BlockSize * Size, DT_MATRIX_MPI, 0, ColComm);
    }
    for (int i=0; i<BlockSize; i++) {
        MPI_Scatter(&pMatrixRow[i*Size], BlockSize, DT_MATRIX_MPI, &(pMatrixBlock[ i * BlockSize]), BlockSize, DT_MATRIX_MPI, 0, RowComm);
    }
    free(pMatrixRow);
}

// Function for data distribution among the processes
void DataDistribution(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, DT_MATRIX* pMatrixAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock, int Size, int BlockSize) {
    CheckerboardMatrixScatter(pAMatrix, pMatrixAblock, Size, BlockSize);
    CheckerboardMatrixScatter(pBMatrix, pBblock, Size, BlockSize);
    CheckerboardMatrixScatter(pCMatrix, pCblock, Size, BlockSize);
    
    //TestBlocks(pCblock, BlockSize, "Block of C matrix");
}

// Function for creating the two-dimensional grid communicator and // communicators for each row and each column of the grid
void CreateGridCommunicators() {
    int DimSize[2];  // Number of processes in each dimension of the grid
    int Periodic[2]; // =1, if the grid dimension should be periodic
    int Subdims[2]; // =1, if the grid dimension should be fixed
    
    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 1;
    Periodic[1] = 1;
    
    // Creation of the Cartesian communicator GridSize x GridSize with "paradox"
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    
    // Determination of the cartesian coordinates for every process
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);
    
    // Createing communicators for rows
    Subdims[0] = 0; // Dimension is fixed
    Subdims[1] = 1; // Dimension belong to the subgrid
    MPI_Cart_sub(GridComm, Subdims, &RowComm);
    
    // Creating communicators for columns
    Subdims[0] = 1; // Dimension belong to the subgrid
    Subdims[1] = 0; // Dimension is fixed
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}


void PrintMatrix(DT_MATRIX *pMatrix, int Size){
    int i, j;  // Loop variables
    for (i=0; i < Size; i++) {
        for (j = 0; j < Size; j++) {
            printf("%ld ", pMatrix[i*Size+j]);
        }
        printf("\n");
    }
}

// Function for matrix multiplication
void SerialResultCalculation(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size) {
    int i, j, k;  // Loop variables
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++) {
            for (k=0; k<Size; k++) {
                
                pCMatrix[i*Size+j] += pAMatrix[i*Size+k] * pBMatrix[k*Size+j];
            }
        }
    }
}

// Function for the special matrix multiplication
void SpecialMatrixMultiply(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size) {
    int i, j, k;  // Loop variables
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++) {
            for (k=0; k<Size; k++) {
                
                pCMatrix[i*Size+j] = minimum(pCMatrix[i*Size+j], pAMatrix[i*Size+k] + pBMatrix[k*Size+j]);
            }
        }
    }
}


// Function for memory allocation and initialization of matrix elements
void ProcessInitialization (DT_MATRIX** pAMatrix, DT_MATRIX** pBMatrix, DT_MATRIX** pCMatrix, DT_MATRIX** pAblock, DT_MATRIX** pBblock, DT_MATRIX** pCblock, DT_MATRIX** pMatrixAblock, int *Size, int *BlockSize, char* input ){
    // Setting the size of matrices
    /*if (ProcRank == 0) {
     do {
     printf("\nEnter the size of matrices:\n ");
     scanf("%d", Size);
     if ((*Size) % GridSize != 0)
     printf("\nSize of matrices must be divisible by the grid size!\n");
     }
     while ((*Size) % GridSize != 0);
     }*/
    
    if (ProcRank == 0){
        FILE * file;
        file = fopen(input, "r");
        
        if (file == NULL)
        {
            printf("Error! Could not open file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        char c[128];
        fscanf(file, "%s", c);
        *Size = atoi(c);
        
        fclose(file);
        
        // Validation
        if ((*Size) % GridSize != 0){
            printf("\nERROR:Size of matrices must be divisible by the grid size!\n");
            printf("ERROR: Invalid configuration!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        //printf("Size %d\n\n\n\n", *Size);
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Allocating the memmory
    *BlockSize = (*Size)/GridSize;
    (*pAblock) = malloc(sizeof(DT_MATRIX) * (*BlockSize) * (*BlockSize));
    (*pBblock) = malloc(sizeof(DT_MATRIX) * (*BlockSize) * (*BlockSize));
    (*pCblock) = malloc(sizeof(DT_MATRIX) * (*BlockSize) * (*BlockSize));
    (*pMatrixAblock) = malloc(sizeof(DT_MATRIX) * (*BlockSize) * (*BlockSize));
    
    if (ProcRank == 0) {
        (*pAMatrix) = malloc(sizeof(DT_MATRIX) * (*Size) * (*Size));
        (*pBMatrix) = malloc(sizeof(DT_MATRIX) * (*Size) * (*Size));
        (*pCMatrix) = malloc(sizeof(DT_MATRIX) * (*Size) * (*Size));
        
        // Initialize the matrices
        DataInitialization(*pAMatrix, *pBMatrix, *pCMatrix, *Size, input);
    }
    
    // Init the result block of each process (c)
    for (int i = 0; i < (*BlockSize) * (*BlockSize); i++) {
        (*pCblock)[i] = 0;
    }
}

void DataInitialization(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size, char * input){
    int i, j;  // Loop variables
    
    FILE * file;
    file = fopen(input, "r");
    
    if (file == NULL)
    {
        printf("Error! Could not open file\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    // reading the size (again) but we don't need it now
    char str[128];
    fscanf(file, "%s", str);
    
    for (i=0; i < Size; i++){
        for (j=0; j<Size; j++) {
            fscanf(file, "%s", str);
            
            DT_MATRIX number = atoi(str);
            if(number == 0 && i != j){
                pAMatrix[i*Size+j] = MAX_MATRIX_VALUE;
                pBMatrix[i*Size+j] = MAX_MATRIX_VALUE;
                pCMatrix[i*Size+j] = MAX_MATRIX_VALUE;
            }
            else {
                pAMatrix[i*Size+j] = number;
                pBMatrix[i*Size+j] = number;
                pCMatrix[i*Size+j] = number;
            }
        }
    }
    fclose(file);
    
    /*if (DEBUG){
     printf("\nMatrix A:\n");
     PrintMatrix(pAMatrix, Size);
     
     printf("\nMatrix B:\n");
     PrintMatrix(pBMatrix, Size);
     }*/
}

void copyArrays(DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, int Size) {
    int i, j;  // Loop variables
    for (i=0; i<Size; i++) {
        for (j=0; j<Size; j++) {
            pAMatrix[i*Size+j] = pCMatrix[i*Size+j];
            pBMatrix[i*Size+j] = pCMatrix[i*Size+j];
        }
    }
}


// Function for computational process termination
void ProcessTermination (DT_MATRIX* pAMatrix, DT_MATRIX* pBMatrix, DT_MATRIX* pCMatrix, DT_MATRIX* pAblock, DT_MATRIX* pBblock, DT_MATRIX* pCblock,DT_MATRIX* pMatrixAblock){
    if (ProcRank == 0) {
        free(pAMatrix);
        free(pBMatrix);
        free(pCMatrix);
    }
    free(pAblock);
    free(pBblock);
    free(pCblock);
    free(pMatrixAblock);
}
