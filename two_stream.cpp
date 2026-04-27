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
static constexpr uint   DEFAULT_NP    = 1u<<12;  // 2^12 ~= 4,000
// static constexpr uint   DEFAULT_NP    = 1u<<20;          // 1,048,576
static const     float  DEFAULT_L     = 4.0f*pif;        // ~12.566; fits one unstable wavelength for k0=0.5
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
	GLFWwindow* win = glfwCreateWindow(1280, 720, "two-stream PIC (OpenCL)", nullptr, nullptr);
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
	bool  show_grid     = false; // overlay vertical lines at each cell boundary
	int   view_mode     = 0;     // 0 = phase space (particles), 1 = electric field E(x)
	float emax_view     = 1.0f;  // y-axis half-range for the E-field plot

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

		// Clear the framebuffer (both view modes share this).
		int win_w, win_h;
		glfwGetFramebufferSize(win, &win_w, &win_h);
		glViewport(0, 0, win_w, win_h);
		glClearColor(0.04f, 0.04f, 0.06f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Plot rectangle in NDC, computed from logical window size so the
		// control panel on the left and label margins are reserved consistently.
		// Used by both the GL particle pass and the ImGui-overlay annotations.
		int wlog_w, wlog_h;
		glfwGetWindowSize(win, &wlog_w, &wlog_h);
		const float panel_w     = 300.0f;             // logical pixels reserved for left panel
		const float plot_left   = panel_w + 56.0f;    // gap for y-axis word + tick text
		const float plot_right  = (float)wlog_w - 16.0f;
		const float plot_top    = 28.0f;              // gap for title text
		const float plot_bottom = (float)wlog_h - 48.0f; // gap for x-axis ticks + label
		const float x0_ndc = -1.0f + 2.0f*plot_left   /(float)wlog_w;
		const float x1_ndc = -1.0f + 2.0f*plot_right  /(float)wlog_w;
		const float y0_ndc =  1.0f - 2.0f*plot_bottom /(float)wlog_h;
		const float y1_ndc =  1.0f - 2.0f*plot_top    /(float)wlog_h;

		// In phase-space mode, upload particles to VBOs and draw them via GL.
		// Field mode draws E(x) via ImGui's draw list below, no GL particles.
		uint Np_draw = 0u;
		if(view_mode==0) {
			Np_draw = (sim.Np+(uint)stride_view-1u)/(uint)stride_view;
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
			glUniform2f(uXRange, x0_ndc, x1_ndc);
			glUniform2f(uYRange, y0_ndc, y1_ndc);
			glUniform1i(uShape, draw_circles ? 1 : 0);
			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, (GLsizei)Np_draw);
			glBindVertexArray(0);
		}

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
		ImGui::Text("View:");
		ImGui::SameLine(); ImGui::RadioButton("phase space", &view_mode, 0);
		ImGui::SameLine(); ImGui::RadioButton("E field",     &view_mode, 1);
		ImGui::Checkbox("Show grid", &show_grid);
		if(view_mode==0) {
			ImGui::SliderFloat("v range", &vmax_view, 0.5f, 5.0f);
			ImGui::SliderInt("Render stride", &stride_view, 1, 32);
			ImGui::Checkbox("Filled circles", &draw_circles);
		} else {
			ImGui::SliderFloat("E range", &emax_view, 0.05f, 5.0f);
		}
		if(ImGui::Button("Reset")) sim.initialize_two_stream();
		ImGui::End();

		// ---- Axis / overlay rendering on top of the plot ----------------
		// Uses the same plot rectangle (x0_ndc..x1_ndc, y0_ndc..y1_ndc) computed
		// at the top of this frame so the labels line up with the GL viewport.
		{
			const ImVec2 disp = ImGui::GetIO().DisplaySize; // logical pixels
			auto ndc_to_px = [&](float nx, float ny) {
				return ImVec2((nx+1.0f)*0.5f*disp.x, (1.0f-ny)*0.5f*disp.y);
			};
			const ImVec2 tl  = ndc_to_px(x0_ndc, y1_ndc); // top-left of plot rect
			const ImVec2 br  = ndc_to_px(x1_ndc, y0_ndc); // bottom-right
			const ImVec2 axL = ndc_to_px(x0_ndc, 0.0f);   // y=0 axis line endpoints
			const ImVec2 axR = ndc_to_px(x1_ndc, 0.0f);
			ImDrawList* dl = ImGui::GetForegroundDrawList();
			const ImU32 col_frame = IM_COL32(220, 220, 220, 220);
			const ImU32 col_axis  = IM_COL32(180, 180, 180, 110);
			const ImU32 col_text  = IM_COL32(230, 230, 235, 255);
			const ImU32 col_dim   = IM_COL32(180, 180, 190, 200);
			const ImU32 col_grid  = IM_COL32(120, 130, 160,  80);
			const ImU32 col_field = IM_COL32(255, 210,  90, 255);

			// Optional grid: vertical line at every cell node x = i*dx for i=0..Nx.
			// Even at Nx=1024 this is a few thousand line vertices for ImGui.
			if(show_grid) {
				const float plot_w = br.x-tl.x;
				for(uint i=0u; i<=sim.Nx; i++) {
					const float fx = (float)i/(float)sim.Nx;
					const float px = tl.x+fx*plot_w;
					dl->AddLine(ImVec2(px, tl.y), ImVec2(px, br.y), col_grid, 1.0f);
				}
			}

			// In field mode, draw the E(x) curve as a polyline. E was already
			// pulled to host this frame by sim.read_state_to_host().
			if(view_mode==1) {
				static vector<ImVec2> pts;
				pts.resize(sim.Nx);
				const float plot_w = br.x-tl.x;
				const float plot_h = br.y-tl.y;
				for(uint i=0u; i<sim.Nx; i++) {
					const float fx = (float)i/(float)sim.Nx;
					float fy = 0.5f-0.5f*sim.E[i]/emax_view; // y flipped: +E is up
					if(fy<0.0f) fy = 0.0f;
					if(fy>1.0f) fy = 1.0f;
					pts[i] = ImVec2(tl.x+fx*plot_w, tl.y+fy*plot_h);
				}
				dl->AddPolyline(pts.data(), (int)pts.size(), col_field, ImDrawFlags_None, 1.5f);
			}

			// Plot frame and y=0 axis line (drawn after grid/curve so they stay on top).
			dl->AddRect(tl, br, col_frame, 0.0f, 0, 1.0f);
			dl->AddLine(axL, axR, col_axis, 1.0f);

			// Mode-dependent labels.
			const char* title    = (view_mode==0) ? "1d1v phase space" : "electric field  E(x)";
			const char* y_word   = (view_mode==0) ? "v  velocity"      : "E  field";
			const float y_range  = (view_mode==0) ? vmax_view          : emax_view;

			char buf[64];
			ImVec2 ts;

			// Title centered above the plot.
			ts = ImGui::CalcTextSize(title);
			dl->AddText(ImVec2(0.5f*(tl.x+br.x)-0.5f*ts.x, tl.y-ts.y-6.0f), col_text, title);

			// X-axis tick labels (0 and L) under the plot, plus axis label "x".
			snprintf(buf, sizeof(buf), "0");
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(tl.x-0.5f*ts.x, br.y+3.0f), col_dim, buf);
			snprintf(buf, sizeof(buf), "L = %.2f", sim.L);
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(br.x-ts.x, br.y+3.0f), col_dim, buf);
			snprintf(buf, sizeof(buf), "x  (position)");
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(0.5f*(tl.x+br.x)-0.5f*ts.x, br.y+20.0f), col_text, buf);

			// Y-axis tick labels and axis label.
			snprintf(buf, sizeof(buf), "+%.2f", y_range);
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(tl.x-ts.x-6.0f, tl.y-0.5f*ts.y), col_dim, buf);
			snprintf(buf, sizeof(buf), "0");
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(tl.x-ts.x-6.0f, axL.y-0.5f*ts.y), col_dim, buf);
			snprintf(buf, sizeof(buf), "-%.2f", y_range);
			ts = ImGui::CalcTextSize(buf);
			dl->AddText(ImVec2(tl.x-ts.x-6.0f, br.y-0.5f*ts.y), col_dim, buf);
			// Stack the y-axis word vertically (ImGui has no rotated text).
			// Place it just to the right of the panel, left of the tick labels.
			const float y_word_x = panel_w + 6.0f;
			float y_cursor = 0.5f*(tl.y+br.y)-0.5f*(float)strlen(y_word)*ImGui::GetFontSize();
			for(const char* p=y_word; *p; p++) {
				char ch[2] = { *p, 0 };
				dl->AddText(ImVec2(y_word_x, y_cursor), col_text, ch);
				y_cursor += ImGui::GetFontSize();
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
