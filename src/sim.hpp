#pragma once

#include <random>
#include "opencl.hpp"
// NOTE: do not include kernel.hpp here — its R(...) stringification macro
// pollutes any subsequent header (e.g. <random>) that uses `R` as an identifier.
// opencl.hpp already forward-declares get_opencl_c_code(); kernel.cpp emits the body.

// One-dimensional, one-velocity electrostatic PIC simulation of the two-stream
// instability. Periodic boundary conditions, electron-only, with a uniform
// neutralizing ion background. FP32 throughout.
//
// Algorithm per step:
//   1. zero cell_count
//   2. k_count: each particle atomic_inc's its cell's count
//   3. k_scan:  exclusive prefix-sum of cell_count -> cell_offset
//   4. zero cell_write_idx
//   5. k_sort:  counting-sort particles into cell-major order (ping-pong)
//   6. k_deposit: per-node CIC charge deposition (gather form, no atomics)
//   7. k_field: cumulative integral of (rho - <rho>) -> E, then subtract <E>
//   8. k_push:  leapfrog kick-drift, periodic wrap
//
// Normalization: omega_p = 1, n_0 = 1, eps_0 = 1, m_e = 1, q_e = -1.

class TwoStreamSim {
public:
	// Parameters (all FP32 host-side too).
	uint Nx;        // number of cells/nodes
	uint Np;        // number of macroparticles
	float L;        // domain length
	float dx;
	float dt;
	float v0;       // beam drift speed
	float vt;       // thermal spread
	float qp;       // charge per macroparticle = -L/Np (electrons)
	float qm;       // q/m = -1 for electrons
	float seed_amp; // amplitude of position perturbation that seeds the instability

	// Diagnostics (host-side, recomputed on demand).
	uint step_count = 0u;
	double t = 0.0;
	float ke = 0.0f, pe = 0.0f;

	// Device + buffers.
	Device device;
	Memory<float> x_a, v_a, x_b, v_b;
	Memory<float> stream_a, stream_b; // per-particle initial-stream tag (0 = right-mover at t=0, 1 = left-mover at t=0); ping-ponged with x and v through the counting sort.
	Memory<float> rho, E;
	Memory<uint> cell_count, cell_offset, cell_write_idx;
	Memory<float> partial; // partial sums for v^2 reduction

	// Kernels.
	Kernel k_zero;
	Kernel k_count;
	Kernel k_scan;
	Kernel k_sort;
	Kernel k_deposit;
	Kernel k_field;
	Kernel k_push;
	Kernel k_reduce_v2;

	int curr = 0; // 0 -> a holds current state, 1 -> b holds current state

	uint reduce_groups; // number of work-groups for v^2 reduction

	inline TwoStreamSim(const Device_Info& dev_info, uint Nx_, uint Np_, float L_, float v0_, float vt_, float dt_, float seed_amp_=0.01f) {
		this->Nx = Nx_;
		this->Np = Np_;
		this->L = L_;
		this->dx = L_/(float)Nx_;
		this->dt = dt_;
		this->v0 = v0_;
		this->vt = vt_;
		this->seed_amp = seed_amp_;
		this->qm = -1.0f;
		this->qp = -L_/(float)Np_;
		this->reduce_groups = (Np_+(uint)WORKGROUP_SIZE-1u)/(uint)WORKGROUP_SIZE;

		const string defines =
			"\n	#define def_Np "+to_string(Np)+"u"
			"\n	#define def_Nx "+to_string(Nx)+"u"
			"\n	#define def_dx "+to_string(dx)+"f"
			"\n	#define def_dt "+to_string(dt)+"f"
			"\n	#define def_L "+to_string(L)+"f"
			"\n	#define def_qm "+to_string(qm)+"f"
			"\n	#define def_qp "+to_string(qp)+"f"
		;
		this->device = Device(dev_info, defines+get_opencl_c_code());

		// Particle buffers (ping-pong).
		this->x_a = Memory<float>(device, Np);
		this->v_a = Memory<float>(device, Np);
		this->x_b = Memory<float>(device, Np);
		this->v_b = Memory<float>(device, Np);
		this->stream_a = Memory<float>(device, Np);
		this->stream_b = Memory<float>(device, Np);

		// Field buffers.
		this->rho = Memory<float>(device, Nx);
		this->E   = Memory<float>(device, Nx);

		// Bin buffers (size Nx+1 so the same zero kernel works for all).
		this->cell_count     = Memory<uint>(device, Nx+1u);
		this->cell_offset    = Memory<uint>(device, Nx+1u);
		this->cell_write_idx = Memory<uint>(device, Nx+1u);

		// Reduction partial sums.
		this->partial = Memory<float>(device, reduce_groups);

		// Construct kernels with initial parameter bindings; we'll rebind ping-pong buffers each step.
		this->k_zero      = Kernel(device, Nx+1u,            "k_zero",      cell_count);
		this->k_count     = Kernel(device, Np,               "k_count",     x_a, cell_count);
		this->k_scan      = Kernel(device, 1ull,             "k_scan",      cell_count, cell_offset);
		this->k_sort      = Kernel(device, Np,               "k_sort",      x_a, v_a, stream_a, x_b, v_b, stream_b, cell_offset, cell_write_idx);
		this->k_deposit   = Kernel(device, Nx,               "k_deposit",   x_b, cell_offset, rho);
		this->k_field     = Kernel(device, 1ull,             "k_field",     rho, E);
		this->k_push      = Kernel(device, Np,               "k_push",      x_b, v_b, E);
		this->k_reduce_v2 = Kernel(device, Np,               "k_reduce_v2", v_b, partial);
	}

	// Pick the "live" particle buffers based on curr.
	inline Memory<float>& x_curr() { return curr==0 ? x_a : x_b; }
	inline Memory<float>& v_curr() { return curr==0 ? v_a : v_b; }
	inline Memory<float>& s_curr() { return curr==0 ? stream_a : stream_b; }
	inline Memory<float>& x_next() { return curr==0 ? x_b : x_a; }
	inline Memory<float>& v_next() { return curr==0 ? v_b : v_a; }
	inline Memory<float>& s_next() { return curr==0 ? stream_b : stream_a; }

	// Two counter-streaming Maxwellians, uniform x with a small sinusoidal seed.
	inline void initialize_two_stream(uint seed=12345u) {
		std::mt19937 rng(seed);
		std::normal_distribution<float> gauss(0.0f, vt);
		const float k0 = 2.0f*pif/L; // one-wavelength perturbation across the box
		Memory<float>& xb = x_curr();
		Memory<float>& vb = v_curr();
		Memory<float>& sb = s_curr();
		// Interleave streams in x so both populations span the whole domain.
		// Even index -> right-mover, odd index -> left-mover. Positions are placed on a
		// uniform grid of Np points so neither stream is spatially clustered.
		for(uint i=0u; i<Np; i++) {
			const bool right_mover = ((i&1u)==0u);
			const float v_mean = right_mover ? +v0 : -v0;
			float xp = ((float)i+0.5f)*L/(float)Np;
			xp += seed_amp*sin(k0*xp);
			xp -= L*floor(xp/L);
			if(xp<0.0f) xp += L;
			if(xp>=L)   xp -= L;
			xb[i] = xp;
			vb[i] = v_mean+gauss(rng);
			sb[i] = right_mover ? 0.0f : 1.0f;
		}
		xb.write_to_device();
		vb.write_to_device();
		sb.write_to_device();
		t = 0.0;
		step_count = 0u;
		curr = 0;
	}

	// One full timestep.
	inline void step() {
		Memory<float>& xs = x_curr();
		Memory<float>& vs = v_curr();
		Memory<float>& ss = s_curr();
		Memory<float>& xd = x_next();
		Memory<float>& vd = v_next();
		Memory<float>& sd = s_next();

		// 1. Zero cell_count.
		k_zero.set_parameters(0u, cell_count).run();
		// 2. Count particles per cell.
		k_count.set_parameters(0u, xs, cell_count).run();
		// 3. Exclusive scan -> cell_offset.
		k_scan.run();
		// 4. Zero cell_write_idx.
		k_zero.set_parameters(0u, cell_write_idx).run();
		// 5. Counting-sort particles (x, v, stream-tag) into the destination buffers.
		k_sort.set_parameters(0u, xs, vs, ss, xd, vd, sd, cell_offset, cell_write_idx).run();
		// 6. CIC deposit from sorted xd into rho.
		k_deposit.set_parameters(0u, xd, cell_offset, rho).run();
		// 7. Solve E from rho.
		k_field.run();
		// 8. Leapfrog push, in place on xd, vd.
		k_push.set_parameters(0u, xd, vd, E).run();

		curr = 1-curr;
		step_count++;
		t += (double)dt;
	}

	// Pull current particle arrays + E to host (no-op when zero-copy).
	inline void read_state_to_host() {
		x_curr().read_from_device();
		v_curr().read_from_device();
		s_curr().read_from_device();
		E.read_from_device();
	}

	// Compute kinetic and potential energy diagnostics from host-side state.
	// Call read_state_to_host() first.
	inline void compute_energies() {
		// PE: (1/2) integral E^2 dx over the box.
		float pe_sum = 0.0f;
		for(uint i=0u; i<Nx; i++) pe_sum += E[i]*E[i];
		pe = 0.5f*dx*pe_sum;

		// KE: GPU tree reduction of v^2, sum partials on host, scale by mass per macroparticle.
		// Mass per macroparticle = L/Np in our normalization (so m_e * n_0 = 1).
		k_reduce_v2.set_parameters(0u, v_curr(), partial).run();
		partial.read_from_device();
		double v2sum = 0.0;
		for(uint i=0u; i<reduce_groups; i++) v2sum += (double)partial[i];
		ke = 0.5f*((float)v2sum)*(L/(float)Np);
	}
};
