// Two-stream electrostatic PIC demo with live phase-space rendering.
//
// OpenCL compute kernels follow the FluidX3D / OpenCL-Benchmark pattern:
// kernels live as a single OpenCL C string compiled at runtime; host code
// uses thin Memory<T>/Kernel/Device wrappers from ../OpenCL-Benchmark/src/.
//
// Rendering: GLFW window, OpenGL 3.2 core, two VBOs (x and v) drawn as GL_POINTS.
// UI overlay via Dear ImGui (auto-fetched into extern/imgui by make.sh).

#include "src/sim.hpp"

#include <GLFW/glfw3.h>
#ifdef __APPLE__
	#include <OpenGL/gl3.h>
#else
	#include <GL/glew.h>
#endif

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

// ---- Default simulation parameters ----------------------------------------
static constexpr uint   DEFAULT_NX    = 1024u;
// static constexpr uint   DEFAULT_NP    = 1u<<14;  // 2^12 ~= 4,000
static constexpr uint   DEFAULT_NP    = 1u<<18;          // 1,048,576
static const     float  DEFAULT_L     = 8.0f*pif;        // ~25.13; mode k=4 sits near peak growth (K=k*v0/omega_p=1.0, gamma~0.49 vs gamma_max=0.5). k=1,2 also unstable; k=8 is at the stability boundary.
static constexpr float  DEFAULT_V0    = 1.0f;
static constexpr float  DEFAULT_VT    = 0.1f;
static constexpr float  DEFAULT_DT    = 0.1f;            // omega_p * dt = 0.1
static constexpr float  DEFAULT_SEED  = 0.05f;           // position perturbation amplitude

// ---- GL helpers -----------------------------------------------------------
static GLuint compile_shader(GLenum type, const char* src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if(!ok) {
		char log[2048];
		glGetShaderInfoLog(s, sizeof(log), nullptr, log);
		std::fprintf(stderr, "Shader compile error: %s\n", log);
	}
	return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if(!ok) {
		char log[2048];
		glGetProgramInfoLog(p, sizeof(log), nullptr, log);
		std::fprintf(stderr, "Program link error: %s\n", log);
	}
	return p;
}

// Vertex shader: maps (x in [0,L], v in [-vmax,vmax]) -> NDC. The stream tag
// (0 or 1, set at t=0 and carried through the sort) flows through to the
// fragment shader for coloring.
static const char* VERT_SRC = R"(
#version 330 core
layout(location=0) in float a_x;
layout(location=1) in float a_v;
layout(location=2) in float a_s;
uniform float u_L;
uniform float u_vmax;
uniform float u_pointsize;
uniform vec2  u_x_range;
uniform vec2  u_y_range;
out float v_s;
void main() {
	float fx = a_x / u_L;
	float fy = clamp(0.5 + 0.5*a_v / u_vmax, 0.0, 1.0);
	float ndc_x = mix(u_x_range.x, u_x_range.y, fx);
	float ndc_y = mix(u_y_range.x, u_y_range.y, fy);
	gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
	gl_PointSize = u_pointsize;
	v_s = a_s;
}
)";

// Fragment shader: color by *initial* stream membership.
//   v_s == 0.0  -> right-moving stream at t=0 (initial v = +v0) -> blue
//   v_s == 1.0  -> left-moving  stream at t=0 (initial v = -v0) -> orange
// u_shape selects the splat shape:
//   0 -> translucent square; alpha < 1 with additive blending so dense
//        regions glow (the host enables additive blending in this mode).
//   1 -> opaque filled circle; pixels outside the unit disk are discarded
//        and alpha = 1 (the host disables blending so each particle draws
//        as a crisp solid disk with no see-through).
static const char* FRAG_SRC = R"(
#version 330 core
in float v_s;
uniform int u_shape;
out vec4 frag;
void main() {
	vec3 c = (v_s < 0.5) ? vec3(0.30, 0.70, 1.00) : vec3(1.00, 0.55, 0.20);
	float a = 0.25;
	if(u_shape == 1) {
		vec2 p = gl_PointCoord - vec2(0.5);
		if(length(p) > 0.5) discard;
		a = 1.0;
	}
	frag = vec4(c, a);
}
)";

// ---- main -----------------------------------------------------------------
int main(int argc, char* argv[]) {
	// ---- OpenCL device selection -----------------------------------------
	const vector<Device_Info> devices = get_devices();
	Device_Info dev_info;
	if(argc>1) dev_info = select_device_with_id((uint)atoi(argv[1]), devices);
	else       dev_info = select_device_with_most_flops(devices);

	// ---- Simulation -------------------------------------------------------
	TwoStreamSim sim(dev_info, DEFAULT_NX, DEFAULT_NP, DEFAULT_L, DEFAULT_V0, DEFAULT_VT, DEFAULT_DT, DEFAULT_SEED);
	sim.initialize_two_stream();

	// ---- GLFW window ------------------------------------------------------
	if(!glfwInit()) {
		std::fprintf(stderr, "glfwInit failed\n");
		return 1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	GLFWwindow* win = glfwCreateWindow(1280, 720, "Two-stream Instability PIC", nullptr, nullptr);
	if(!win) {
		std::fprintf(stderr, "glfwCreateWindow failed\n");
		glfwTerminate();
		return 1;
	}
	glfwMakeContextCurrent(win);
	glfwSwapInterval(1); // vsync

#ifndef __APPLE__
	glewExperimental = GL_TRUE;
	if(glewInit()!=GLEW_OK) {
		std::fprintf(stderr, "glewInit failed\n");
		return 1;
	}
#endif

	// ---- ImGui ------------------------------------------------------------
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(win, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	// ---- GL objects -------------------------------------------------------
	GLuint vs = compile_shader(GL_VERTEX_SHADER,   VERT_SRC);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
	GLuint prog = link_program(vs, fs);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLuint vao = 0, vbo_x = 0, vbo_v = 0, vbo_s = 0;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo_x);
	glGenBuffers(1, &vbo_v);
	glGenBuffers(1, &vbo_s);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
	glBufferData(GL_ARRAY_BUFFER, sim.Np*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_v);
	glBufferData(GL_ARRAY_BUFFER, sim.Np*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_s);
	glBufferData(GL_ARRAY_BUFFER, sim.Np*sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glBindVertexArray(0);

	const GLint uL         = glGetUniformLocation(prog, "u_L");
	const GLint uVmax      = glGetUniformLocation(prog, "u_vmax");
	const GLint uPointsize = glGetUniformLocation(prog, "u_pointsize");
	const GLint uXRange    = glGetUniformLocation(prog, "u_x_range");
	const GLint uYRange    = glGetUniformLocation(prog, "u_y_range");
	const GLint uShape     = glGetUniformLocation(prog, "u_shape");

	// Point size scales as sqrt(reference_Np / Np) so total drawn area on screen
	// is roughly invariant in Np, keeping the phase-space density visualization
	// recognizable across very different particle counts. Capped to avoid huge
	// blobs at tiny Np.
	const float reference_np = 1024.0f*1024.0f;
	float pointsize = sqrtf(reference_np/(float)sim.Np);
	if(pointsize<1.0f)  pointsize = 1.0f;
	if(pointsize>32.0f) pointsize = 32.0f;

	glEnable(GL_PROGRAM_POINT_SIZE); // required on macOS core profile; harmless elsewhere
	// Blending mode is selected per-frame from draw_circles below.

	// ---- UI state ---------------------------------------------------------
	bool  paused        = true;
	int   substeps      = 1;
	float vmax_view     = 8.0f;
	int   stride_view   = 1; // render every Nth particle (1 = all)
	bool  draw_circles  = true;  // true = opaque filled circle (default), false = translucent square (additive)
	bool  show_grid     = false; // overlay vertical lines at each cell boundary on the phase-space plot
	float emax_view     = 1.0f;  // y-axis half-range for the E(x) plot
	float fmax_view     = 0.0f;  // y-axis upper limit for the f(v) histogram; 0 = auto-scale
	constexpr int hist_bins = 64;

	// Energy time-series, appended once per non-paused frame and cleared by Reset.
	// Running min/max drive auto-scaling of the energy plot's y-axis.
	bool  energy_log = true;
	vector<float> hist_t, hist_ke, hist_pe, hist_te;
	float energy_max_tracked     = 0.0f;
	float energy_min_pos_tracked = INFINITY;

	// Wavenumber of the sinusoidal seed used by Reset: 1 = one wavelength across
	// the box, 2 = two, etc. Linear theory predicts modes with k*v0/omega_p < 1
	// to be unstable, so for v0=1, L=4*pi only k=1 is clearly inside the unstable
	// band — k=2,4,8 are educational comparisons.
	int init_k = 1;

	// ---- Diagnostics bookkeeping -----------------------------------------
	double last_diag_t = glfwGetTime();
	int    diag_steps  = 0;
	float  steps_per_s = 0.0f;
	float  fps         = 0.0f;
	double last_fps_t  = glfwGetTime();
	int    frames      = 0;

	vector<float> draw_tmp;

	// ---- Main loop --------------------------------------------------------
	while(!glfwWindowShouldClose(win)) {
		glfwPollEvents();

		if(!paused) {
			for(int i=0; i<substeps; i++) sim.step();
			diag_steps += substeps;
		}

		sim.read_state_to_host();
		sim.compute_energies();

		// Append a sample to the energy time-series. Only when stepping —
		// otherwise we'd accumulate duplicate points while the user inspects
		// the paused state.
		if(!paused) {
			hist_t .push_back((float)sim.t);
			hist_ke.push_back(sim.ke);
			hist_pe.push_back(sim.pe);
			const float te = sim.ke+sim.pe;
			hist_te.push_back(te);
			if(sim.ke>energy_max_tracked) energy_max_tracked = sim.ke;
			if(sim.pe>energy_max_tracked) energy_max_tracked = sim.pe;
			if(te    >energy_max_tracked) energy_max_tracked = te;
			if(sim.ke>0.0f && sim.ke<energy_min_pos_tracked) energy_min_pos_tracked = sim.ke;
			if(sim.pe>0.0f && sim.pe<energy_min_pos_tracked) energy_min_pos_tracked = sim.pe;
			if(te    >0.0f && te    <energy_min_pos_tracked) energy_min_pos_tracked = te;
		}

		// Clear the framebuffer (all panels share this).
		int win_w, win_h;
		glfwGetFramebufferSize(win, &win_w, &win_h);
		glViewport(0, 0, win_w, win_h);
		glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// 2x2 grid of plot rectangles in logical pixels:
		//   phase space (top-left, large)    f(v)  (top-right, small)
		//   energy      (bottom-left, large) E(x)  (bottom-right, small)
		// Everything else (control panel, axis-text margins) is reserved space.
		int wlog_w, wlog_h;
		glfwGetWindowSize(win, &wlog_w, &wlog_h);
		const float panel_w   = 300.0f; // logical pixels reserved for left control panel
		const float gap_yword = 56.0f;  // space at left of each plot for vertical y-axis word + tick numerals
		const float gap_top   = 28.0f;  // space above each plot for its title
		const float gap_bot   = 48.0f;  // space below each plot for x-tick labels + axis word
		const float gap_right = 16.0f;

		// Right column takes ~35% of the area after the control panel; rows split evenly.
		const float right_col_w = 0.35f*((float)wlog_w-panel_w);
		const float right_col_x = (float)wlog_w-right_col_w;
		const float mid_y       = 0.5f*(float)wlog_h;

		const float ps_left  = panel_w     + gap_yword;
		const float ps_right = right_col_x - gap_right;
		const float ps_top   = gap_top;
		const float ps_bot   = mid_y - gap_bot;

		const float en_left  = panel_w     + gap_yword;
		const float en_right = right_col_x - gap_right;
		const float en_top   = mid_y + gap_top;
		const float en_bot   = (float)wlog_h - gap_bot;

		const float f_left   = right_col_x + gap_yword;
		const float f_right  = (float)wlog_w - gap_right;
		const float f_top    = gap_top;
		const float f_bot    = mid_y - gap_bot;

		const float e_left   = right_col_x + gap_yword;
		const float e_right  = (float)wlog_w - gap_right;
		const float e_top    = mid_y + gap_top;
		const float e_bot    = (float)wlog_h - gap_bot;

		// Phase-space rect in NDC (used by the GL particle pass).
		const float ps_x0_ndc = -1.0f + 2.0f*ps_left /(float)wlog_w;
		const float ps_x1_ndc = -1.0f + 2.0f*ps_right/(float)wlog_w;
		const float ps_y0_ndc =  1.0f - 2.0f*ps_bot  /(float)wlog_h;
		const float ps_y1_ndc =  1.0f - 2.0f*ps_top  /(float)wlog_h;

		// Upload particles to VBOs and draw the phase-space scatter via GL.
		const uint Np_draw = (sim.Np+(uint)stride_view-1u)/(uint)stride_view;
		glBindBuffer(GL_ARRAY_BUFFER, vbo_x);
		if(stride_view==1) {
			glBufferSubData(GL_ARRAY_BUFFER, 0, sim.Np*sizeof(float), sim.x_curr().data());
		} else {
			draw_tmp.resize(Np_draw);
			const float* src = sim.x_curr().data();
			for(uint i=0u; i<Np_draw; i++) draw_tmp[i] = src[i*(uint)stride_view];
			glBufferSubData(GL_ARRAY_BUFFER, 0, Np_draw*sizeof(float), draw_tmp.data());
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo_v);
		if(stride_view==1) {
			glBufferSubData(GL_ARRAY_BUFFER, 0, sim.Np*sizeof(float), sim.v_curr().data());
		} else {
			draw_tmp.resize(Np_draw);
			const float* src = sim.v_curr().data();
			for(uint i=0u; i<Np_draw; i++) draw_tmp[i] = src[i*(uint)stride_view];
			glBufferSubData(GL_ARRAY_BUFFER, 0, Np_draw*sizeof(float), draw_tmp.data());
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo_s);
		if(stride_view==1) {
			glBufferSubData(GL_ARRAY_BUFFER, 0, sim.Np*sizeof(float), sim.s_curr().data());
		} else {
			draw_tmp.resize(Np_draw);
			const float* src = sim.s_curr().data();
			for(uint i=0u; i<Np_draw; i++) draw_tmp[i] = src[i*(uint)stride_view];
			glBufferSubData(GL_ARRAY_BUFFER, 0, Np_draw*sizeof(float), draw_tmp.data());
		}

		// Opaque circles: no blending — alpha=1 is meaningful only when each
		// fragment fully replaces what's beneath. Translucent squares: additive
		// blending so overlapping particles brighten dense regions.
		if(draw_circles) {
			glDisable(GL_BLEND);
		} else {
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		}

		glUseProgram(prog);
		glUniform1f(uL, sim.L);
		glUniform1f(uVmax, vmax_view);
		glUniform1f(uPointsize, pointsize);
		glUniform2f(uXRange, ps_x0_ndc, ps_x1_ndc);
		glUniform2f(uYRange, ps_y0_ndc, ps_y1_ndc);
		glUniform1i(uShape, draw_circles ? 1 : 0);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, (GLsizei)Np_draw);
		glBindVertexArray(0);

		// ---- ImGui overlay -----------------------------------------------
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Keyboard shortcuts. Skip when a future ImGui text widget would want the keys.
		if(!ImGui::GetIO().WantCaptureKeyboard) {
			if(ImGui::IsKeyPressed(ImGuiKey_Space, false)) paused = !paused;
		}

		// Always-visible left-side control panel: pinned to (0,0), full window
		// height, with no move/resize/collapse so it never overlaps the plot.
		ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(panel_w, (float)wlog_h), ImGuiCond_Always);
		ImGui::Begin("Two-stream PIC", nullptr,
			ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
		ImGui::Text("Device: %s", sim.device.info.name.c_str());
		ImGui::Text("Np=%u  Nx=%u  L=%.3f  dt=%.3f", sim.Np, sim.Nx, sim.L, sim.dt);
		ImGui::Separator();
		ImGui::Text("step  = %u", sim.step_count);
		ImGui::Text("t     = %.3f  (omega_p * t)", sim.t);
		ImGui::Text("KE    = %.5f", sim.ke);
		ImGui::Text("PE    = %.5f", sim.pe);
		ImGui::Text("KE+PE = %.5f", sim.ke+sim.pe);
		ImGui::Text("FPS   = %.1f", fps);
		ImGui::Text("steps/s = %.0f", steps_per_s);
		ImGui::Separator();
		ImGui::Checkbox("Paused (space)", &paused);
		ImGui::SliderInt("Substeps/frame", &substeps, 1, 50);

		ImGui::Separator();
		ImGui::TextUnformatted("Phase space");
		ImGui::SliderFloat("v range",     &vmax_view, 0.5f, 5.0f);
		ImGui::SliderInt("Render stride", &stride_view, 1, 32);
		ImGui::Checkbox("Filled circles", &draw_circles);
		ImGui::Checkbox("Show grid",      &show_grid);
		ImGui::Separator();
		ImGui::TextUnformatted("E field");
		ImGui::SliderFloat("E range",     &emax_view, 0.05f, 5.0f);
		ImGui::Separator();
		ImGui::TextUnformatted("f(v)");
		ImGui::SliderFloat("f range",     &fmax_view, 0.0f, 5.0f);
		ImGui::TextDisabled("(0 = auto-scale)");
		ImGui::Separator();
		ImGui::TextUnformatted("Energy");
		ImGui::Checkbox("Log scale##E",   &energy_log);
		ImGui::Separator();
		ImGui::TextUnformatted("Initial perturbation (on Reset)");
		ImGui::RadioButton("k=1", &init_k, 1); ImGui::SameLine();
		ImGui::RadioButton("k=2", &init_k, 2); ImGui::SameLine();
		// ImGui::RadioButton("k=4", &init_k, 4); ImGui::SameLine();
		ImGui::RadioButton("k=4", &init_k, 4);
		if(ImGui::Button("Reset")) {
			sim.initialize_two_stream((uint)init_k);
			hist_t.clear();
			hist_ke.clear();
			hist_pe.clear();
			hist_te.clear();
			energy_max_tracked     = 0.0f;
			energy_min_pos_tracked = INFINITY;
		}
		ImGui::End();

		// ---- Axis / overlay rendering on top of the three plots ----------
		{
			ImDrawList* dl = ImGui::GetForegroundDrawList();
			const ImU32 col_frame   = IM_COL32(220, 220, 220, 220);
			const ImU32 col_axis    = IM_COL32(180, 180, 180, 110);
			const ImU32 col_text    = IM_COL32(230, 230, 235, 255);
			const ImU32 col_dim     = IM_COL32(180, 180, 190, 200);
			const ImU32 col_grid    = IM_COL32(120, 130, 160,  80);
			const ImU32 col_field   = IM_COL32(255, 210,  90, 255);
			// Histogram trace colors must match FRAG_SRC's per-stream colors so the
			// f(v) plot's blue/orange line up with the phase-space dot colors.
			const ImU32 col_stream0 = IM_COL32( 76, 178, 255, 255); // right-mover (blue)
			const ImU32 col_stream1 = IM_COL32(255, 140,  51, 255); // left-mover  (orange)

			// Frame + title + axis ticks + axis words for one rectangular plot.
			// `yt_mid` is null when the y range doesn't cross zero (f(v) is 0..fmax);
			// when non-null it adds a horizontal axis line at the rect midpoint and a
			// "0" tick label there.
			auto draw_plot_frame = [&](float x0, float y0, float x1, float y1,
			                           const char* title,
			                           const char* x_word, const char* xt_lo, const char* xt_hi,
			                           const char* y_word, const char* yt_lo, const char* yt_hi,
			                           const char* yt_mid) {
				const ImVec2 tl(x0, y0);
				const ImVec2 br(x1, y1);
				dl->AddRect(tl, br, col_frame, 0.0f, 0, 1.0f);
				if(yt_mid) {
					const float ymid = 0.5f*(tl.y+br.y);
					dl->AddLine(ImVec2(tl.x, ymid), ImVec2(br.x, ymid), col_axis, 1.0f);
				}
				ImVec2 ts;
				ts = ImGui::CalcTextSize(title);
				dl->AddText(ImVec2(0.5f*(tl.x+br.x)-0.5f*ts.x, tl.y-ts.y-6.0f), col_text, title);
				ts = ImGui::CalcTextSize(xt_lo);
				dl->AddText(ImVec2(tl.x-0.5f*ts.x, br.y+3.0f), col_dim, xt_lo);
				ts = ImGui::CalcTextSize(xt_hi);
				dl->AddText(ImVec2(br.x-ts.x, br.y+3.0f), col_dim, xt_hi);
				ts = ImGui::CalcTextSize(x_word);
				dl->AddText(ImVec2(0.5f*(tl.x+br.x)-0.5f*ts.x, br.y+20.0f), col_text, x_word);
				ts = ImGui::CalcTextSize(yt_hi);
				dl->AddText(ImVec2(tl.x-ts.x-6.0f, tl.y-0.5f*ts.y), col_dim, yt_hi);
				ts = ImGui::CalcTextSize(yt_lo);
				dl->AddText(ImVec2(tl.x-ts.x-6.0f, br.y-0.5f*ts.y), col_dim, yt_lo);
				if(yt_mid) {
					ts = ImGui::CalcTextSize(yt_mid);
					const float ymid = 0.5f*(tl.y+br.y);
					dl->AddText(ImVec2(tl.x-ts.x-6.0f, ymid-0.5f*ts.y), col_dim, yt_mid);
				}
				// Y-axis word, stacked vertically (ImGui has no rotated text).
				const float y_word_x = tl.x-50.0f;
				float y_cursor = 0.5f*(tl.y+br.y)-0.5f*(float)strlen(y_word)*ImGui::GetFontSize();
				for(const char* p=y_word; *p; p++) {
					char ch[2] = { *p, 0 };
					dl->AddText(ImVec2(y_word_x, y_cursor), col_text, ch);
					y_cursor += ImGui::GetFontSize();
				}
			};

			char xt_hi_buf[32], yt_hi_buf[32], yt_lo_buf[32];

			// ---- Phase-space plot (particles drawn via GL above; only annotations here) ----
			if(show_grid) {
				const float plot_w = ps_right-ps_left;
				for(uint i=0u; i<=sim.Nx; i++) {
					const float fx = (float)i/(float)sim.Nx;
					const float px = ps_left+fx*plot_w;
					dl->AddLine(ImVec2(px, ps_top), ImVec2(px, ps_bot), col_grid, 1.0f);
				}
			}
			snprintf(xt_hi_buf, sizeof(xt_hi_buf), "L = %.2f", sim.L);
			snprintf(yt_hi_buf, sizeof(yt_hi_buf), "+%.2f", vmax_view);
			snprintf(yt_lo_buf, sizeof(yt_lo_buf), "-%.2f", vmax_view);
			draw_plot_frame(ps_left, ps_top, ps_right, ps_bot,
				"1d1v phase space",
				"x  (position)", "0",       xt_hi_buf,
				"v  velocity",   yt_lo_buf, yt_hi_buf, "0");

			// ---- E(x) plot ----
			{
				static vector<ImVec2> pts;
				pts.resize(sim.Nx);
				const float plot_w = e_right-e_left;
				const float plot_h = e_bot-e_top;
				for(uint i=0u; i<sim.Nx; i++) {
					const float fx = (float)i/(float)sim.Nx;
					float fy = 0.5f-0.5f*sim.E[i]/emax_view; // y flipped: +E is up
					if(fy<0.0f) fy = 0.0f;
					if(fy>1.0f) fy = 1.0f;
					pts[i] = ImVec2(e_left+fx*plot_w, e_top+fy*plot_h);
				}
				dl->AddPolyline(pts.data(), (int)pts.size(), col_field, ImDrawFlags_None, 1.5f);
			}
			snprintf(xt_hi_buf, sizeof(xt_hi_buf), "L = %.2f", sim.L);
			snprintf(yt_hi_buf, sizeof(yt_hi_buf), "+%.2f", emax_view);
			snprintf(yt_lo_buf, sizeof(yt_lo_buf), "-%.2f", emax_view);
			draw_plot_frame(e_left, e_top, e_right, e_bot,
				"electric field  E(x)",
				"x  (position)", "0",       xt_hi_buf,
				"E  field",      yt_lo_buf, yt_hi_buf, "0");

			// ---- f(v) histogram, two stepped curves (one per initial-stream tag) ----
			{
				// Bin v into hist_bins over [-vmax_view, +vmax_view], split by initial stream.
				static vector<int> hist0, hist1;
				hist0.assign(hist_bins, 0);
				hist1.assign(hist_bins, 0);
				const float* vp = sim.v_curr().data();
				const float* sp = sim.s_curr().data();
				const float scale = (float)hist_bins/(2.0f*vmax_view);
				for(uint i=0u; i<sim.Np; i++) {
					const float fx = (vp[i]+vmax_view)*scale;
					if(fx<0.0f || fx>=(float)hist_bins) continue; // off the plot, drop it
					const int bin = (int)fx;
					if(sp[i]<0.5f) hist0[bin]++;
					else           hist1[bin]++;
				}
				// PDF normalization: dividing count_per_bin by Np*dv makes the *total*
				// f integrate to 1 over v; each stream alone integrates to ~0.5 since
				// it's half the population.
				const float dv = 2.0f*vmax_view/(float)hist_bins;
				const float pdf_norm = 1.0f/((float)sim.Np*dv);

				float fmax_eff = fmax_view;
				if(fmax_eff<=0.0f) {
					float m = 0.0f;
					for(int b=0; b<hist_bins; b++) {
						const float f0 = (float)hist0[b]*pdf_norm;
						const float f1 = (float)hist1[b]*pdf_norm;
						if(f0>m) m = f0;
						if(f1>m) m = f1;
					}
					fmax_eff = m>0.0f ? 1.1f*m : 1.0f;
				}

				const float plot_w = f_right-f_left;
				const float plot_h = f_bot-f_top;
				// Stepped polyline: 2 vertices per bin (left edge + right edge at the
				// same height) -> flat top across each bin, vertical drops between bins.
				static vector<ImVec2> pts0, pts1;
				pts0.resize(2*(size_t)hist_bins);
				pts1.resize(2*(size_t)hist_bins);
				for(int b=0; b<hist_bins; b++) {
					const float fxL = (float)b/(float)hist_bins;
					const float fxR = (float)(b+1)/(float)hist_bins;
					const float xL = f_left+fxL*plot_w;
					const float xR = f_left+fxR*plot_w;
					const float f0 = (float)hist0[b]*pdf_norm;
					const float f1 = (float)hist1[b]*pdf_norm;
					float y0 = f_bot-(f0/fmax_eff)*plot_h;
					float y1 = f_bot-(f1/fmax_eff)*plot_h;
					if(y0<f_top) y0 = f_top;
					if(y1<f_top) y1 = f_top;
					pts0[2*b]   = ImVec2(xL, y0);
					pts0[2*b+1] = ImVec2(xR, y0);
					pts1[2*b]   = ImVec2(xL, y1);
					pts1[2*b+1] = ImVec2(xR, y1);
				}
				dl->AddPolyline(pts0.data(), (int)pts0.size(), col_stream0, ImDrawFlags_None, 1.5f);
				dl->AddPolyline(pts1.data(), (int)pts1.size(), col_stream1, ImDrawFlags_None, 1.5f);

				snprintf(xt_hi_buf, sizeof(xt_hi_buf), "+%.2f", vmax_view);
				snprintf(yt_lo_buf, sizeof(yt_lo_buf), "-%.2f", vmax_view); // x-axis low tick (reused buffer)
				snprintf(yt_hi_buf, sizeof(yt_hi_buf), "%.3f", fmax_eff);
				draw_plot_frame(f_left, f_top, f_right, f_bot,
					"distribution  f(v)",
					"v  velocity", yt_lo_buf, xt_hi_buf,
					"f",           "0",       yt_hi_buf, nullptr);
			}

			// ---- Energy plot: KE, PE, KE+PE vs t (log y by default) ----
			{
				const ImU32 col_ke = IM_COL32( 80, 160, 255, 255); // blue
				const ImU32 col_pe = IM_COL32(120, 220, 100, 255); // green
				const ImU32 col_te = IM_COL32(255,  90,  90, 255); // red

				// Y-axis range. In log mode, snap to whole decades just outside the
				// observed min/max; in linear mode, 0..1.1*max. Both are sticky against
				// the running min/max so the axis doesn't jitter as new extremes appear.
				float y_lo, y_hi;
				if(energy_log) {
					const float lo_v = energy_min_pos_tracked<INFINITY ? energy_min_pos_tracked : 1e-12f;
					const float hi_v = energy_max_tracked>0.0f         ? energy_max_tracked     : 1.0f;
					y_lo = floorf(log10f(lo_v));
					y_hi = ceilf (log10f(hi_v));
					if(y_hi-y_lo<2.0f) y_lo = y_hi-2.0f;
				} else {
					y_lo = 0.0f;
					y_hi = energy_max_tracked>0.0f ? 1.1f*energy_max_tracked : 1.0f;
				}
				const float t_max = (!hist_t.empty() && hist_t.back()>1.0f) ? hist_t.back() : 1.0f;

				const float plot_w = en_right-en_left;
				const float plot_h = en_bot-en_top;
				const int   N      = (int)hist_t.size();

				if(N>=2) {
					// Decimate to ~one vertex per pixel column to keep ImGui draw cost bounded
					// even for very long runs.
					const int max_vertices = (int)plot_w>2 ? (int)plot_w : 2;
					const int stride = N>max_vertices ? N/max_vertices : 1;

					auto build_curve = [&](const vector<float>& vals, vector<ImVec2>& pts) {
						pts.clear();
						pts.reserve((size_t)((N+stride-1)/stride));
						for(int i=0; i<N; i+=stride) {
							const float fx = hist_t[i]/t_max;
							float fy;
							if(energy_log) {
								const float lv = log10f(vals[i]>1e-30f ? vals[i] : 1e-30f);
								fy = (lv-y_lo)/(y_hi-y_lo);
							} else {
								fy = (vals[i]-y_lo)/(y_hi-y_lo);
							}
							if(fy<0.0f) fy = 0.0f;
							if(fy>1.0f) fy = 1.0f;
							pts.push_back(ImVec2(en_left+fx*plot_w, en_bot-fy*plot_h));
						}
					};

					static vector<ImVec2> pts_ke, pts_pe, pts_te;
					build_curve(hist_ke, pts_ke);
					build_curve(hist_pe, pts_pe);
					build_curve(hist_te, pts_te);
					if((int)pts_ke.size()>=2) dl->AddPolyline(pts_ke.data(), (int)pts_ke.size(), col_ke, ImDrawFlags_None, 1.5f);
					if((int)pts_pe.size()>=2) dl->AddPolyline(pts_pe.data(), (int)pts_pe.size(), col_pe, ImDrawFlags_None, 1.5f);
					if((int)pts_te.size()>=2) dl->AddPolyline(pts_te.data(), (int)pts_te.size(), col_te, ImDrawFlags_None, 1.5f);
				}

				// Inline legend: small color swatches + names in the top-left of the plot.
				{
					float lx = en_left+8.0f;
					float ly = en_top +6.0f;
					auto entry = [&](ImU32 col, const char* label) {
						dl->AddRectFilled(ImVec2(lx, ly+3.0f), ImVec2(lx+14.0f, ly+13.0f), col);
						dl->AddText(ImVec2(lx+20.0f, ly), col_text, label);
						ly += 16.0f;
					};
					entry(col_ke, "KE");
					entry(col_pe, "PE");
					entry(col_te, "KE+PE");
				}

				char xt_hi_e[32], yt_hi_e[32], yt_lo_e[32];
				snprintf(xt_hi_e, sizeof(xt_hi_e), "%.1f", t_max);
				if(energy_log) {
					snprintf(yt_hi_e, sizeof(yt_hi_e), "1e%+d", (int)y_hi);
					snprintf(yt_lo_e, sizeof(yt_lo_e), "1e%+d", (int)y_lo);
				} else {
					snprintf(yt_hi_e, sizeof(yt_hi_e), "%.2f", y_hi);
					snprintf(yt_lo_e, sizeof(yt_lo_e), "%.2f", y_lo);
				}
				draw_plot_frame(en_left, en_top, en_right, en_bot,
					"energy",
					"t  (omega_p t)", "0",      xt_hi_e,
					"energy",         yt_lo_e,  yt_hi_e, nullptr);
			}
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(win);

		// Update steps/s and FPS counters every ~0.5 s.
		const double now = glfwGetTime();
		const double diag_dt = now-last_diag_t;
		if(diag_dt>=0.5) {
			steps_per_s = (float)((double)diag_steps/diag_dt);
			last_diag_t = now;
			diag_steps = 0;
		}
		frames++;
		if(now-last_fps_t>=0.5) {
			fps = (float)((double)frames/(now-last_fps_t));
			frames = 0;
			last_fps_t = now;
		}
	}

	// ---- Cleanup ----------------------------------------------------------
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glDeleteProgram(prog);
	glDeleteBuffers(1, &vbo_x);
	glDeleteBuffers(1, &vbo_v);
	glDeleteVertexArrays(1, &vao);
	glfwDestroyWindow(win);
	glfwTerminate();
	return 0;
}
