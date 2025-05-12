/* Program to sample from the Bayesian posterior distribution of the DCSBM
 * using Monte Carlo and perform link prediction
 *
 * Written by Mark Newman  6 MAY 2024
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
#include <gsl/gsl_sf_gamma.h>

#include "bipartite.h"

/* Constants */
#define CROSSV 0.1             // Fraction of edges to remove
#define K 100                  // Maximum number of groups on either side
#define MCSWEEPS 75000         // Number of Monte Carlo sweeps to perform
#define EQUIL 50000            // Equilibration time
#define SAMPLE 10              // Rate at which to sample results, in sweeps
#define CHECKINT 1000          // Interval at which to update output, in sweeps
#define LINELENGTH 10000

/* Globals */
BIPARTITE G;          // Struct storing the network
int nnodes;
int nedges;           // Number of edges
double p;             // Probability of an edge

int k[2];             // Current value of k for each type
int *g[2];            // Group assignments for each type
int *n[2];            // Group sizes for each type
int **m;              // Edge counts
int *kappa[2];        // Sums of degrees for each type
int **in[2];          // Lists of the nodes in each group

int nsamples=0;       // Number of samples drawn
double **predict;     // Prediction probabilities
int **removed;        // Which edges were removed

double *lnfact;       // Look-up table of log-factorials
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
  nnodes = G.n[0] + G.n[1];
  p = (double)nedges/(G.n[0]*G.n[1]);

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


// Make a lookup table of log-factorial values
void maketable()
{
  int t;
  int length;

  length = nnodes + nedges + 1;
  lnfact = malloc(length*sizeof(double));
  for (t=0; t<length; t++) lnfact[t] = gsl_sf_lnfact(t);
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


// Initial group assignment
void initgroups()
{
  int i,j,u,v;
  int r;
  int *perm;

  // Make space for the group labels
  g[0] = malloc(G.n[0]*sizeof(int));
  g[1] = malloc(G.n[1]*sizeof(int));

  // For nodes of each type, place one randomly chosen node in each group,
  // then put the remaining nodes in random groups.  Works by generating a
  // random permutation of the integers 1...n then placing nodes in groups in
  // that order.

  // Type 0
  perm = malloc(G.n[0]*sizeof(int));
  for (i=0; i<G.n[0]; i++) {
    j = gsl_rng_uniform_int(rng,i+1);
    perm[i] = perm[j];
    perm[j] = i;
  }
  for (i=0; i<K; i++) g[0][perm[i]] = i;
  for (; i<G.n[0]; i++) g[0][perm[i]] = gsl_rng_uniform_int(rng,K);
  free(perm);

  // Calculate the n's, kappas, and the lists of group members
  n[0] = calloc(K,sizeof(int));
  kappa[0] = calloc(K,sizeof(int));
  in[0] = malloc(K*sizeof(int*));
  for (r=0; r<K; r++) in[0][r] = malloc(G.n[0]*sizeof(int));
  for (u=0; u<G.n[0]; u++) {
    r = g[0][u];
    in[0][r][n[0][r]] = u;
    n[0][r] += 1;
    kappa[0][r] += G.node[0][u].degree;
  }

  // Initialize k0
  k[0] = K;

  // Now do the same for type 1
  perm = malloc(G.n[1]*sizeof(int));
  for (i=0; i<G.n[1]; i++) {
    j = gsl_rng_uniform_int(rng,i+1);
    perm[i] = perm[j];
    perm[j] = i;
  }
  for (i=0; i<K; i++) g[1][perm[i]] = i;
  for (; i<G.n[1]; i++) g[1][perm[i]] = gsl_rng_uniform_int(rng,K);
  free(perm);

  n[1] = calloc(K,sizeof(int));
  kappa[1] = calloc(K,sizeof(int));
  in[1] = malloc(K*sizeof(int*));
  for (r=0; r<K; r++) in[1][r] = malloc(G.n[1]*sizeof(int));
  for (u=0; u<G.n[1]; u++) {
    r = g[1][u];
    in[1][r][n[1][r]] = u;
    n[1][r] += 1;
    kappa[1][r] += G.node[1][u].degree;
  }

  k[1] = K;

  // Calculate the values of the m's
  m = malloc(K*sizeof(int*));
  for (r=0; r<K; r++) m[r] = calloc(K,sizeof(int));
  for (u=0; u<G.n[0]; u++) {
    for (i=0; i<G.node[0][u].degree; i++) {
      v = G.node[0][u].edge[i].target;
      m[g[0][u]][g[1][v]]++;
    }
  }
}


// Function to update n, kappa, and m for a proposed move
void nmupdate(int type, int u, int d[], int r, int s)
{
  int t;

  n[type][r]--;
  n[type][s]++;
  kappa[type][r] -= G.node[type][u].degree;
  kappa[type][s] += G.node[type][u].degree;
  if (type==0) {
    for (t=0; t<k[1]; t++) {
      m[r][t] -= d[t];
      m[s][t] += d[t];
    }
  } else {
    for (t=0; t<k[0]; t++) {
      m[t][r] -= d[t];
      m[t][s] += d[t];
    }
  }
}


// Function to calculate the change in the log-likelihood upon moving one
// node (Type A move)
double deltalogpA(int type, int u, int d[], int r, int s)
{
  int t;
  int nr,ns,nrp,nsp;
  int kappar,kappas,kapparp,kappasp;
  int mrt,mst,mrtp,mstp;
  int nt;
  double res;

  if (n[type][r]==1) {

    // Calculate the n's and kappa's
    ns = n[type][s];
    nsp = n[type][s] + 1;
    kappar = kappa[type][r];
    kappas = kappa[type][s];
    kappasp = kappa[type][s] + G.node[type][u].degree;

    // Leading terms
    res = lnfact[kappar] + kappasp*log(nsp) - (kappas-1)*log(ns)
          + lnfact[ns+kappas-1] - lnfact[nsp+kappasp-1];

    // Now all the terms in the sum
    if (type==0) {
      for (t=0; t<k[1]; t++) {
	nt = n[1][t];
	mrt = m[r][t];
	mst = m[s][t];
	mstp = m[s][t] + d[t];
	res += (mrt+1)*log(p*nt+1) - lnfact[mrt]
               + (mst+1)*log(p*ns*nt+1) - (mstp+1)*log(p*nsp*nt+1);
               - lnfact[mst] + lnfact[mstp];
      }
    } else {
      for (t=0; t<k[0]; t++) {
	nt = n[0][t];
	mrt = m[t][r];
	mst = m[t][s];
	mstp = m[t][s] + d[t];
	res += (mrt+1)*log(p*nt+1) - lnfact[mrt]
               + (mst+1)*log(p*ns*nt+1) - (mstp+1)*log(p*nsp*nt+1);
               - lnfact[mst] + lnfact[mstp];
      }
    }

  } else {

    // Calculate the n's and kappa's
    nr = n[type][r];
    ns = n[type][s];
    nrp = n[type][r] - 1;
    nsp = n[type][s] + 1;
    kappar = kappa[type][r];
    kappas = kappa[type][s];
    kapparp = kappa[type][r] - G.node[type][u].degree;
    kappasp = kappa[type][s] + G.node[type][u].degree;

    // Leading terms
    res = (kapparp-1)*log(nrp) - kappar*log(nr)
          + lnfact[nr+kappar-1] - lnfact[nrp+kapparp-1]
          + kappasp*log(nsp) - (kappas-1)*log(ns)
          + lnfact[ns+kappas-1] - lnfact[nsp+kappasp-1];

    // Now all the terms in the sum
    if (type==0) {
      for (t=0; t<k[1]; t++) {
	nt = n[1][t];
	mrt = m[r][t];
	mrtp = m[r][t] - d[t];
	mst = m[s][t];
	mstp = m[s][t] + d[t];
	res += lnfact[mrtp] - lnfact[mrt]
	       + (mrt+1)*log(p*nr*nt+1) - (mrtp+1)*log(p*nrp*nt+1)
               + lnfact[mstp] - lnfact[mst]
               + (mst+1)*log(p*ns*nt+1) - (mstp+1)*log(p*nsp*nt+1);
      }
    } else {
      for (t=0; t<k[0]; t++) {
	nt = n[0][t];
	mrt = m[t][r];
	mrtp = m[t][r] - d[t];
	mst = m[t][s];
	mstp = m[t][s] + d[t];
	res += lnfact[mrtp] - lnfact[mrt]
	       + (mrt+1)*log(p*nr*nt+1) - (mrtp+1)*log(p*nrp*nt+1)
               + lnfact[mstp] - lnfact[mst]
               + (mst+1)*log(p*ns*nt+1) - (mstp+1)*log(p*nsp*nt+1);
      }
    }
  }

  return res;
}
          

// Function to calculate the change in log-likelihood upon creating a
// new group (Type B move)
double deltalogpB(int type, int u, int d[], int r)
{
  int t;
  int nt,nr,nrp;
  int kappar,kapparp;
  int mrt,mrtp;
  double res;

  // Calculate the new n_r and kappa_r
  nr = n[type][r];
  nrp = n[type][r] - 1;
  kappar = kappa[type][r];
  kapparp = kappa[type][r] - G.node[type][u].degree;
  
  // Leading terms
  res = (kapparp-1)*log(nrp) - kappar*log(nr)
        + lnfact[nr+kappar-1] - lnfact[nrp+kapparp-1]
        - lnfact[G.node[type][u].degree];

  // Terms in the sum
  if (type==0) {
    for (t=0; t<k[1]; t++) {
      nt = n[1][t];
      mrt = m[r][t];
      mrtp = m[r][t] - d[t];
      res += lnfact[mrtp] - lnfact[mrt]
             + (mrt+1)*log(p*nr*nt+1) - (mrtp+1)*log(p*nrp*nt+1)
	     + lnfact[d[t]] - (d[t]+1)*log(p*nt+1);
    }
  } else {
    for (t=0; t<k[0]; t++) {
      nt = n[0][t];
      mrt = m[t][r];
      mrtp = m[t][r] - d[t];
      res += lnfact[mrtp] - lnfact[mrt]
             + (mrt+1)*log(p*nr*nt+1) - (mrtp+1)*log(p*nrp*nt+1)
	     + lnfact[d[t]] - (d[t]+1)*log(p*nt+1);
    }
  }

  return res;
}


// Function that does one Monte Carlo sweep (i.e., n individual moves)
double sweep()
{
  int i,j,u,v;
  int r,s,t;
  int qr,type;
  int accept=0;
  int d[K];

  for (i=0; i<nnodes; i++) {

    // Choose the type of node we are going to move
    type = gsl_rng_uniform_int(rng,2);

    // Choose the type of move we are going to do
    if (gsl_rng_uniform_int(rng,G.n[type]-1)>0) {

      // Type A: move a node
      if (k[type]==1) continue;       // No moves are possible when k=1

      // Choose two distinct groups at random
      r = gsl_rng_uniform_int(rng,k[type]);
      s = gsl_rng_uniform_int(rng,k[type]-1);
      if (r==s) s = k[type]-1;

      // Choose a node u at random from group r
      qr = gsl_rng_uniform_int(rng,n[type][r]);
      u = in[type][r][qr];

      // Count the number of edges this node has to each group of the
      // other type
      for (t=0; t<k[1-type]; t++) d[t] = 0;
      for (j=0; j<G.node[type][u].degree; j++) {
	v = G.node[type][u].edge[j].target;
	d[g[1-type][v]]++;
      }

      // Decide whether to accept the move
      if (gsl_rng_uniform(rng)<exp(deltalogpA(type,u,d,r,s))) {
	g[type][u] = s;
	in[type][r][qr] = in[type][r][n[type][r]-1];
	in[type][s][n[type][s]] = u;
	nmupdate(type,u,d,r,s);
	accept++;

	// If we have removed the last node from group r,
        // decrease k and relabel
	if (n[type][r]==0) {

	  k[type]--;
	  if (r!=k[type]) {

	    // Update the group labels and the lists
	    for (qr=0; qr<n[type][k[type]]; qr++) {
	      v = in[type][r][qr] = in[type][k[type]][qr];
	      g[type][v] = r;
	    }

	    // Update n_r and kappa_r
	    n[type][r] = n[type][k[type]];
	    n[type][k[type]] = 0;
	    kappa[type][r] = kappa[type][k[type]];
	    kappa[type][k[type]] = 0;

	    // Update m_rs
	    if (type==0) {
	      for (t=0; t<k[1]; t++) {
		m[r][t] = m[k[0]][t];
		m[k[0]][t] = 0;
	      }
	    } else {
	      for (t=0; t<k[0]; t++) {
		m[t][r] = m[t][k[1]];
		m[t][k[1]] = 0;
	      }
	    }
	  }
	}
      }

    } else {

      // Type 2: move a node to a newly created group
      if (k[type]==K) continue;    // No room to increase k, so do nothing

      // Choose source group at random
      r = gsl_rng_uniform_int(rng,k[type]);
      if (n[type][r]==1) continue;  // Moving last node in group does nothing

      // Choose a node at random in group r
      qr = gsl_rng_uniform_int(rng,n[type][r]);
      u = in[type][r][qr];

      // Count the number of edges this node has to each group of other type
      for (t=0; t<k[1-type]; t++) d[t] = 0;
      for (j=0; j<G.node[type][u].degree; j++) {
	v = G.node[type][u].edge[j].target;
	d[g[1-type][v]]++;
      }
      
      // Decide whether to accept the move
      if (gsl_rng_uniform(rng)<exp(deltalogpB(type,u,d,r))) {
	g[type][u] = k[type];
	in[type][r][qr] = in[type][r][n[type][r]-1];
	in[type][k[type]][0] = u;
        nmupdate(type,u,d,r,k[type]);
	k[type]++;
	accept++;
      }
    }
  }

  return (double)accept/nnodes;
}


// Sample the relative probabilities of adding every possible edge
void sample()
{
  int i,j,k;
  int r,s;
  double prob;

  for (i=0; i<G.n[0]; i++) {
    for (j=0; j<G.n[1]; j++) {
      r = g[0][i];
      s = g[1][j];
      prob = (double)(m[r][s]+1)/(n[0][r]*n[1][s]+1/p)
             *(G.node[0][i].degree+1)*(G.node[1][j].degree+1)/(G.n[0]*G.n[1])
	     *n[0][r]*n[1][s]/(n[0][r]+kappa[0][r])*(n[1][s]+kappa[1][s]);
      predict[i][j] += prob;
    }
  }
  nsamples++;
}      


main(int argc, char *argv[])
{
  int i,j,u;
  int exists;
  double accept;
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

  // Make lookup table
  maketable();

  // Initialize the group assignment
  initgroups();

  // Make space for the prediction probabilities
  predict = malloc(G.n[0]*sizeof(double*));
  for (i=0; i<G.n[0]; i++) predict[i] = calloc(G.n[1],sizeof(double));

  // Perform the Monte Carlo
  for (i=0; i<=MCSWEEPS; i++) {

    // Perform one MC sweep
    accept = sweep();

    // Samples
    if (i%SAMPLE==0) {
      if (i>=EQUIL) sample();
    }

    // Updates
    if (i%CHECKINT==0) printf("%i %i %i %g\n",i,k[0],k[1],accept);
  }

  // Write out the results
  sprintf(filename,"dcsbm-%06i.txt",seed);
  fp = fopen(filename,"w");
  for (i=0; i<G.n[0]; i++) {
    for (j=0; j<G.n[1]; j++) {
      for (u=exists=0; u<G.node[0][i].degree; u++) {
	if (G.node[0][i].edge[u].target==j) {
	  exists = 1;
	  break;
	}
      }
      if (!exists) {
	fprintf(fp,"%i %i %g %i\n",i,j,predict[i][j]/nsamples,removed[i][j]);
      }
    }
  }
  fclose(fp);
}
