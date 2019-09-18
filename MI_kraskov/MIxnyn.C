// Copyright 2009 Alexander Kraskov, Harald Stoegbauer, Peter Grassberger
//-----------------------------------------------------------------------------------------
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should receive a copy of the GNU General Public License
// along with this program.  See also <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------------------
// Contacts:
//
// Harald Stoegbauer <h.stoegbauer@gmail.com>
// Alexander Kraskov <alexander.kraskov@gmail.com>
//-----------------------------------------------------------------------------------------
// Please reference
//
// A. Kraskov, H. Stogbauer, and P. Grassberger,
// Estimating mutual information.
// Phys. Rev. E 69 (6) 066138, 2004
//
// in your published research.

// 2019 Naoyuki Kamo: Modified to create Python-binding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "miutils.h"

double MIxnyn(double *_x, int dimx, int dimy, int N, int K, double addnoise) {

  int i;
  double **x;
  double *scal;
  double *min;
  double *max;
  double *psi;
  int d;
  double mir;

  int BOX1;

  double s,me;

  x=(double**)calloc(dimx+dimy,sizeof(double*));
  // for (d=0;d<dimx+dimy;d++) x[d]=(double*)calloc(N,sizeof(double));
  scal=(double*)calloc(dimx+dimy,sizeof(double));
  min=(double*)calloc(dimx+dimy,sizeof(double));
  max=(double*)calloc(dimx+dimy,sizeof(double));
  for (d=0;d<dimx+dimy;d++) {min[d]=DBL_MAX/2;max[d]=-DBL_MAX/2;}
  // Copying input data
  for (d=0;d<dimx+dimy;d++) {
      x[d] = _x + N * d;
  }
  // add noise
  if (addnoise) {
    srand((dimx+dimy)*N*K*(int)(x[(dimx+dimy)/2][N/10]));
    if (addnoise==-1) for (d=0;d<dimx+dimy;d++) for (i=0;i<N;i++) x[d][i]+=(1.0*rand()/RAND_MAX)*1e-8;
    else for (d=0;d<dimx+dimy;d++) for (i=0;i<N;i++) x[d][i]+=(1.0*rand()/RAND_MAX)*addnoise;
  }

  //normalization
  for (d=0;d<dimx+dimy;d++) {
    me=s=0; for (i=0;i<N;i++) me+=x[d][i];
    me/=N;  for (i=0;i<N;i++) s+=(x[d][i]-me)*(x[d][i]-me);
    s/=(N-1);s=sqrt(s);
    if (s==0) {;}
    for (i=0;i<N;i++) {
      x[d][i] = (x[d][i]-me)/s;
      if (x[d][i]<min[d]) min[d]=x[d][i];
      if (x[d][i]>max[d]) max[d]=x[d][i];
    }
    for (i=0;i<N;i++) x[d][i]=x[d][i]-min[d];
  }

  psi=(double*)calloc(N+1,sizeof(double));
  psi[1]=-(double).57721566490153;
  for (i=1;i<N;i++) psi[i+1]=psi[i]+1/(double)i;
  BOX1=N-5;
  for (d=0;d<dimx+dimy;d++) scal[d]=BOX1/(max[d]-min[d]);

  mir_xnyn(x,dimx,dimy,N,K,psi,scal,&mir);

  // for (d=0;d<dimx+dimy;d++) free(x[d]); free(x);
  free(x);
  free(scal);
  free(min);free(max);
  free(psi);
  return mir;
}
