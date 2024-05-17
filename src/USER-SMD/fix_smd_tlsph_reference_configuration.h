/* -*- c++ -*- ----------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * This file is based on the FixShearHistory class.
 *
 * ----------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */


#ifdef FIX_CLASS

FixStyle(SMD_TLSPH_NEIGHBORS,FixSMD_TLSPH_ReferenceConfiguration)

#else

#ifndef LMP_FIX_SMD_TLSPH_REFERENCE_H
#define LMP_FIX_SMD_TLSPH_REFERENCE_H

#include "fix.h"
#include "my_page.h"
#include <Eigen/Eigen>

namespace LAMMPS_NS {

class FixSMD_TLSPH_ReferenceConfiguration: public Fix {
	friend class Neighbor;
	friend class PairTlsph;

public:
	FixSMD_TLSPH_ReferenceConfiguration(class LAMMPS *, int, char **);
	~FixSMD_TLSPH_ReferenceConfiguration();
	int setmask();
	void init();
	void setup(int);
	void pre_exchange();
        void setup_post_neighbor();
        void post_neighbor();
	int pack_forward_comm(int, int *, double *, int, int *);
	void unpack_forward_comm(int, int, double *);

	double memory_usage();
	void grow_arrays(int);
	void copy_arrays(int, int, int);
	int pack_exchange(int, double *);
	int unpack_exchange(int, double *);
	int pack_restart(int, double *);
	void unpack_restart(int, int);
	int size_restart(int);
	int maxsize_restart();

	bool crack_exclude(int i, int j);
	bool get_line_intersection(int i, int j);

	void write_restart(FILE *);
	void restart(char *);
	void forward_comm_tl();

protected:
	int updateFlag; // flag to update reference configuration
	int nmax;
	int maxpartner;
	int *npartner;                // # of touching partners of each atom
	tagint **partner;             // global atom IDs for the partners
	float **wfd_list, **wf_list, **energy_per_bond;
        double **r0, **K_g_dot_dx0_normalized;
	float **degradation_ij; // per-pair interaction degradation status
	Eigen::Vector3d **partnerdx, **g_list; // x0 vector of all partners
	Eigen::Matrix3d *K;
	class Pair *pair;

	int *nrecv;                // # of particles received
	int *nsendlist;            // # of particles to send to other CPUs
	int **sendlist;            // list of particles to be sent to other CPUs
	int *maxsendlist;          // max # of memory slots allocated for sendlist
	double *buf_send;          // send buffer
	double *buf_recv;          // recv buffer
	int maxsend;               // max # of memory slots allocated for buf_send
	int maxrecv;               // max # of memory slots allocated for buf_recv
	int firstrecv;
	int n_missing_all;
	bool need_forward_comm;
	int nprocs;

};

}

#endif
#endif

