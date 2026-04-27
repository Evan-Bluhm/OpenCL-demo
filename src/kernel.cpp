#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



// ===== Two-stream PIC kernels =====
// Defines injected from host:
//   def_Np  (uint)   number of particles
//   def_Nx  (uint)   number of grid cells/nodes
//   def_dx  (float)  cell width
//   def_dt  (float)  timestep
//   def_L   (float)  domain length
//   def_qm  (float)  q/m for the species (-1 for electrons)
//   def_qp  (float)  charge per macroparticle



// Zero a uint buffer of length def_Nx+1 (used for cell_count and cell_write_idx).
kernel void k_zero(global uint* a) {
	const uint i = get_global_id(0);
	if(i<def_Nx+1u) a[i] = 0u;
}



// Compute each particle's cell index and atomically increment cell_count[c].
kernel void k_count(global const float* x, global uint* cell_count) {
	const uint p = get_global_id(0);
	if(p>=def_Np) return;
	const float xp = x[p];
	uint c = (uint)(xp*(1.0f/def_dx));
	if(c>=def_Nx) c = def_Nx-1u;
	atomic_inc(&cell_count[c]);
}



// Exclusive prefix sum over Nx cells; cell_offset[Nx] holds the total particle count.
// Single work-item: range is rounded up to a workgroup; only thread 0 does work.
kernel void k_scan(global const uint* cell_count, global uint* cell_offset) {
	if(get_global_id(0)!=0u) return;
	uint sum = 0u;
	for(uint i=0u; i<def_Nx; i++) {
		cell_offset[i] = sum;
		sum += cell_count[i];
	}
	cell_offset[def_Nx] = sum;
}



// Counting-sort placement: each particle is written to its sorted slot.
// The per-particle stream tag (s_in/s_out) is moved alongside x and v so it
// remains attached to its particle across re-binnings.
kernel void k_sort(
	global const float* x_in, global const float* v_in, global const float* s_in,
	global float* x_out, global float* v_out, global float* s_out,
	global const uint* cell_offset, global uint* cell_write_idx
) {
	const uint p = get_global_id(0);
	if(p>=def_Np) return;
	const float xp = x_in[p];
	const float vp = v_in[p];
	const float sp = s_in[p];
	uint c = (uint)(xp*(1.0f/def_dx));
	if(c>=def_Nx) c = def_Nx-1u;
	const uint dst = cell_offset[c]+atomic_inc(&cell_write_idx[c]);
	x_out[dst] = xp;
	v_out[dst] = vp;
	s_out[dst] = sp;
}



// CIC charge deposition, gather form: one work-item per node n sums contributions
// from the two cells whose particles can touch node n (cell n-1 with weight frac,
// and cell n with weight 1-frac), where frac = xp/dx - cell_index. Periodic
// wraparound on cell n-1 is handled by modulo. Stored as charge density.
kernel void k_deposit(global const float* x, global const uint* cell_offset, global float* rho) {
	const uint n = get_global_id(0);
	if(n>=def_Nx) return;
	const uint cm = (n+def_Nx-1u)%def_Nx; // cell to the "left" of node n (periodic)
	const uint cn = n;                    // cell at node n
	float sum = 0.0f;
	const uint cm_end = cell_offset[cm+1u];
	for(uint i=cell_offset[cm]; i<cm_end; i++) {
		const float xp = x[i];
		const float frac = xp*(1.0f/def_dx)-(float)cm;
		sum += def_qp*frac;
	}
	const uint cn_end = cell_offset[cn+1u];
	for(uint i=cell_offset[cn]; i<cn_end; i++) {
		const float xp = x[i];
		const float frac = xp*(1.0f/def_dx)-(float)cn;
		sum += def_qp*(1.0f-frac);
	}
	rho[n] = sum*(1.0f/def_dx); // charge per unit length
}



// Solve 1D periodic Poisson by direct cumulative integration:
//   dE/dx = rho_neutralized = rho - <rho>     (background ions cancel the mean)
//   E(x)  = integral from 0 to x of rho_neutralized
// Then subtract <E> so the periodic constraint <E>=0 is satisfied.
// Single work-item kernel; runs once per step.
kernel void k_field(global const float* rho, global float* E) {
	if(get_global_id(0)!=0u) return;
	float rho_mean = 0.0f;
	for(uint i=0u; i<def_Nx; i++) rho_mean += rho[i];
	rho_mean *= (1.0f/(float)def_Nx);
	float E_acc = 0.0f;
	float E_sum = 0.0f;
	for(uint i=0u; i<def_Nx; i++) {
		E[i] = E_acc;
		E_acc += (rho[i]-rho_mean)*def_dx;
		E_sum += E[i];
	}
	const float E_mean = E_sum*(1.0f/(float)def_Nx);
	for(uint i=0u; i<def_Nx; i++) E[i] -= E_mean;
}



// Leapfrog push: v_{n+1/2} = v_{n-1/2} + (q/m) E(x_n) dt; x_{n+1} = x_n + v_{n+1/2} dt.
// E is gathered to particle position via CIC. Periodic wrap on x.
kernel void k_push(global float* x, global float* v, global const float* E) {
	const uint p = get_global_id(0);
	if(p>=def_Np) return;
	float xp = x[p];
	float vp = v[p];
	const float fp = xp*(1.0f/def_dx);
	uint i = (uint)fp;
	if(i>=def_Nx) i = def_Nx-1u;
	const float frac = fp-(float)i;
	uint i1 = i+1u;
	if(i1>=def_Nx) i1 = 0u;
	const float Ep = E[i]*(1.0f-frac)+E[i1]*frac;
	vp += def_qm*Ep*def_dt;
	xp += vp*def_dt;
	xp -= def_L*floor(xp*(1.0f/def_L));
	x[p] = xp;
	v[p] = vp;
}



// Tree-reduction of squared velocities for kinetic-energy diagnostic.
// Each work-group reduces WG_SIZE values into one partial sum in partial[group_id].
// Host (or a follow-up tiny kernel) sums the partials.
kernel void k_reduce_v2(global const float* v, global float* partial) {
	local float scratch[cl_workgroup_size];
	const uint lid = get_local_id(0);
	const uint gid = get_global_id(0);
	const float vp = (gid<def_Np) ? v[gid] : 0.0f;
	scratch[lid] = vp*vp;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(uint stride=cl_workgroup_size/2u; stride>0u; stride>>=1) {
		if(lid<stride) scratch[lid] += scratch[lid+stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0u) partial[get_group_id(0)] = scratch[0];
}



);} // ############################################################### end of OpenCL C code #####################################################################
