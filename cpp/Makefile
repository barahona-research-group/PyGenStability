#TODO SIMPLIFY COMPILING/AUTOMATE WITH VARIABLES ETC...

#Define Compiler -- adapt if you are not using gcc/g++
CC=g++-8

#Compilation flags
CFLAGS=-O3 -std=c++0x -Wall

all: run_gen_louvain
	
run_gen_louvain:
	$(CC) $(CFLAGS) run_gen_louvain.cpp -o run_gen_louvain.sh

clean:
	rm *.sh 
#*.o
