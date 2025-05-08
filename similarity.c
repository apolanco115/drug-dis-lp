/* Program to do link prediction using various similarity metrics and
 * then run cross-validation on them
 *
 * Written by Mark Newman  16 MAY 2024
 */

/* Program control */
#define VERBOSE

/* Inclusions */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>

#include "bipartite.h"

/* Constants */
#define CROSSV 0.1             // Fraction of edges to remove
#define LINELENGTH 10000
#define EPSILON 1e-6

/* Macros */
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

/* Globals */
BIPARTITE G;          // Struct storing the network
int nedges;           // Number of edges

double **predict;     // Prediction probabilities
int **removed;        // Which edges were removed

gsl_rng *rng;         // Random number generator


/* Function to read one line from a specified stream.  Return value is
 * 1 if an EOF was encountered.  Otherwise zero */
int readline(FILE *stream, char line[LINELENGTH])
{
  if (fgets(line,LINELENGTH,stream)==NULL) return 1;
  line[strlen(line)-1] = '\0';   /* Erase the terminating NEWLINE */
  return 0;
}


// Read the network
void read_network(FILE *stream)
{
  int i,j;
  int u,v;
  int d;
  int inc;
  char * list;
  char line[LINELENGTH];
  char label[LINELENGTH];

  // Read the numbers of nodes of each type and number of edges
  readline(stream,line);
  sscanf(line,"%i %i %i",&G.n[0],&G.n[1],&nedges);

  // Make space for the nodes
  G.node[0] = malloc(G.n[0]*sizeof(NODE));
  G.node[1] = malloc(G.n[1]*sizeof(NODE));

  // Read in the nodes of type 0
  for (i=0; i<G.n[0]; i++) {
    readline(stream,line);

    // Read node ID, name, and degree
    sscanf(line,"%i %s %i",&u,label,&d);
    G.node[0][u].degree = d;
    strcpy(G.node[0][u].label,label);

    // Make space for the edges then read them in
    G.node[0][u].edge = malloc(d*sizeof(EDGE));
    list = strchr(line,'[') + 1;
    for (j=0; j<d; j++) {
      sscanf(list,"%i%n",&G.node[0][u].edge[j].target,&inc);
      list += inc;
    }
  }

  // Read in the nodes of type 1
  for (i=0; i<G.n[1]; i++) {
    readline(stream,line);

    // Read node ID, name, and degree
    sscanf(line,"%i %s %i",&u,label,&d);
    G.node[1][u].degree = d;
    strcpy(G.node[1][u].label,label);

    // Make space for the edges then read them in
    G.node[1][u].edge = malloc(d*sizeof(EDGE));
    list = strchr(line,'[') + 1;
    for (j=0; j<d; j++) {
      sscanf(list,"%i%n",&G.node[1][u].edge[j].target,&inc);
      list += inc;
    }
  }

#ifdef VERBOSE
  fprintf(stderr,"Read in %i nodes of type 0 and %i of type 1, and %i edges\n",
	  G.n[0],G.n[1],nedges);
#endif
}


// Function to print out the network data
void write_network()
{
  int i,j;

  for (i=0; i<G.n[0]; i++) {
    printf("%i %s %i [ ",i,G.node[0][i].label,G.node[0][i].degree);
    for (j=0; j<G.node[0][i].degree; j++) {
      printf("%i ",G.node[0][i].edge[j].target);
    }
    printf("]\n");
  }

  for (i=0; i<G.n[1]; i++) {
    printf("%i %s %i [ ",i,G.node[1][i].label,G.node[1][i].degree);
    for (j=0; j<G.node[1][i].degree; j++) {
      printf("%i ",G.node[1][i].edge[j].target);
    }
    printf("]\n");
  }
}


// Function to remove edges and save them in a separate array
void remedges()
{
  int i,j,k,l;
  int d;
  int count=0;

  // Make array to record which ones are missing
  removed = malloc(G.n[0]*sizeof(int*));
  for (i=0; i<G.n[0]; i++) removed[i] = calloc(G.n[1],sizeof(int));

  // Go through the edges one by one and choose which ones to remove
  for (i=0; i<G.n[0]; i++) {
    d = G.node[0][i].degree;
    for (k=0; k<d; k++) {

      while ((gsl_rng_uniform(rng)<CROSSV)&&(d>k)) {

	count++;

	// Remove
	j = G.node[0][i].edge[k].target;
	removed[i][j] = 1;
	G.node[0][i].edge[k] = G.node[0][i].edge[--d];

	// Also remove it from the other side
	for (l=0; l<G.node[1][j].degree; l++) {
	  if (G.node[1][j].edge[l].target==i) {
	    G.node[1][j].edge[l] = G.node[1][j].edge[--G.node[1][j].degree];
	    break;
	  }
	}
      }

    }
    G.node[0][i].degree = d;
  }

#ifdef VERBOSE
  fprintf(stderr,"%i edges removed for cross-validation\n",count);
#endif
}


// Calculate prediction scores for all node pairs
void lp()
{
  int u,v,w;
  int i,j;
  int nu,nv;
  int **paths;
  double denom;
  double **sim;

  // Make space
  paths = malloc(G.n[0]*sizeof(int*));
  for (u=0; u<G.n[0]; u++) paths[u] = calloc(G.n[0],sizeof(int));
  sim = malloc(G.n[0]*sizeof(double*));
  for (u=0; u<G.n[0]; u++) sim[u] = malloc(G.n[0]*sizeof(double));

  // Count paths of length two between all pairs of drugs
  for (u=0; u<G.n[0]; u++) {
    for (i=0; i<G.node[0][u].degree; i++) {
      v = G.node[0][u].edge[i].target;
      for (j=0; j<G.node[1][v].degree; j++) {
	w = G.node[1][v].edge[j].target;
	paths[u][w]++;
      }
    }
  }

  // Calculate similarities
  for (u=0; u<G.n[0]; u++) {
    for (v=0; v<G.n[0]; v++) {

      // Calculate denominator

      // Common neighbor count
      denom = 1.0;

      // Cosine similarity
      // denom = sqrt(paths[u][u]*paths[v][v]);

      // Jaccard coefficient
      // denom = paths[u][u]+paths[v][v]-paths[u][v];

      // Dice coefficient
      // denom = 0.5*(paths[u][u]+paths[v][v]);

      // Hub-suppressed
      // denom = MAX(paths[u][u],paths[v][v]);

      if (denom<EPSILON) sim[u][v] = 0.0;
      else sim[u][v] = paths[u][v]/denom;

    }
  }

  // Calculate prediction scores
  for (u=0; u<G.n[0]; u++) {
    for (v=0; v<G.n[1]; v++) {
      predict[u][v] = 0.0;
      for (i=0; i<G.node[1][v].degree; i++) {
	w = G.node[1][v].edge[i].target;
	predict[u][v] += sim[u][w];
      }
    }
  }
}


main(int argc, char *argv[])
{
  int i,u,v;
  int exists;
  char filename[LINELENGTH];
  FILE *fp;
  struct timeval tv;
  unsigned int seed;

  // Initialize the RNG
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gettimeofday(&tv,0);
  seed = tv.tv_usec;
  gsl_rng_set(rng,seed);

  // Read the network from stdin
  read_network(stdin);

  // Remove edges for cross-validation
  remedges();

  // Calculate the prediction scores
  predict = malloc(G.n[0]*sizeof(double*));
  for (u=0; u<G.n[0]; u++) predict[u] = malloc(G.n[1]*sizeof(double));
  lp();

  // Write out the results
  sprintf(filename,"similarity-%06i.txt",seed);
  fp = fopen(filename,"w");
  for (u=0; u<G.n[0]; u++) {
    for (v=0; v<G.n[1]; v++) {
      for (i=exists=0; i<G.node[0][u].degree; i++) {
	if (G.node[0][u].edge[i].target==v) {
	  exists = 1;
	  break;
	}
      }
      if (!exists) fprintf(fp,"%i %i %g %i\n",u,v,predict[u][v],removed[u][v]);
    }
  }
  fclose(fp);
}
