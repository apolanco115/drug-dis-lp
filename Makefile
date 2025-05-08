CFLAGS = -w -O2
CC = gcc
LIBS = -lm -lgsl -lgslcblas

degree:degree.o
	$(CC) $(CFLAGS) -o degree degree.o $(LIBS)

degree.o:degree.c bipartite.h Makefile
	$(CC) $(CFLAGS) -c degree.c

similarity:similarity.o
	$(CC) $(CFLAGS) -o similarity similarity.o $(LIBS)

similarity.o:similarity.c bipartite.h Makefile
	$(CC) $(CFLAGS) -c similarity.c

dcsbm:dcsbm.o
	$(CC) $(CFLAGS) -o dcsbm dcsbm.o $(LIBS)

dcsbm.o:dcsbm.c bipartite.h Makefile
	$(CC) $(CFLAGS) -c dcsbm.c
