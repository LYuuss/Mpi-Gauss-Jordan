CC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11
TARGET = gauss_jordan_mpi
SRC = src/gauss_jordan_mpi.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) data/system3.txt

clean:
	rm -f $(TARGET)

.PHONY: all run clean