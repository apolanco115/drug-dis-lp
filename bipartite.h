// Header file for NODE, EDGE, and BIPARTITE data structures
//
// Mark Newman  6 MAY 2024

#ifndef _BIPARTITE_H
#define _BIPARTITE_H

#define LABELLEN 100

typedef struct {
  int target;        // Index in the node[] array of neighboring node
  double weight;     // Weight of edge (for weighted networks)
} EDGE;

typedef struct {
  int degree;             // Degree of node
  char label[LABELLEN];   // Node label
  EDGE *edge;             // Array of EDGE structs, one for each neighbor
} NODE;

typedef struct {
  int n[2];          // Number of nodes of each type
  NODE *node[2];     // Arrays of NODE structs, one for each node
} BIPARTITE;

#endif
