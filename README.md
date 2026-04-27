# OpenCL-demo

Self-contained OpenCL demos, heavily based on [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) and the [OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark) demo . Compute kernels live as a single OpenCL C string compiled at runtime; rendering is GLFW + OpenGL 3.3 core + [Dear ImGui](https://github.com/ocornut/imgui).

Currently one demo: **`two_stream`** — a 1D-1V electrostatic particle-in-cell simulation of the two-stream instability with live phase-space rendering.

## Build

System dependencies (one-time):

- macOS:  `brew install glfw`
- Linux:  `sudo apt install libglfw3-dev libglew-dev ocl-icd-libopencl1 ocl-icd-opencl-dev`

Then:

```
make             # build bin/two_stream
make run         # build and run
make run ARGS=0  # run on OpenCL device 0 (omit ARGS to auto-pick the highest-FLOPS device)
make clean       # remove app objects and binary
make distclean   # also remove the cached ImGui static archive
```

Dear ImGui (`v1.91.0`) is auto-fetched into `extern/imgui/` on first build and compiled once into `extern/imgui/build/libimgui.a`, so subsequent incremental builds are fast.

## Running

```
./bin/two_stream [device_id]
```

With no argument the runtime picks the device with the highest reported FLOPS, as estimated by OpenCL by multiplying the clock frequency by the maximum number of parallel compute unites. Pass an integer to select a specific OpenCL device.

### Controls

| Key / widget        | Action                                              |
| ------------------- | --------------------------------------------------- |
| `Space`             | Pause / unpause                                     |
| `Substeps/frame`    | How many sim steps to advance per rendered frame    |
| `View`              | Phase space (x, v) or electric field E(x)           |
| `Show grid`         | Overlay the physical compute grid                   |
| `Filled circles`    | Opaque circle particles vs translucent square splats |
| `v range` / `E range` | Vertical scale of the active view                 |
| `Render stride`     | Draw every Nth particle (perf knob, large Np)       |
| `Reset`             | Reinitialize particles                              |

Particles are colored by **initial** stream membership: blue = right-mover at t=0, orange = left-mover at t=0. As the instability grows, phase-space mixing makes the two colors interleave inside the rolls.

## Layout

```
demo-apps/
├── two_stream.cpp         main: GL/ImGui rendering + UI loop
├── src/
│   ├── sim.hpp            TwoStreamSim host class (buffers, step pipeline, diagnostics)
│   ├── kernel.cpp         all OpenCL C kernels, in a single R(...) string
│   ├── kernel.hpp         R(...) stringification macro + editor stubs
│   ├── opencl.hpp         FluidX3D-style Device/Memory/Kernel wrappers
│   ├── utilities.hpp      shared helpers
│   └── OpenCL/            Khronos OpenCL headers (vendored for portability)
├── extern/imgui/          fetched at build time (gitignored)
├── Makefile
├── compile_flags.txt      clangd include paths
└── README.md
```

## Algorithm (per step)

1. Zero `cell_count`.
2. `k_count`: each particle atomic-increments its cell's count.
3. `k_scan`: exclusive prefix-sum → `cell_offset`.
4. Zero `cell_write_idx`.
5. `k_sort`: counting-sort particles into cell-major order (ping-pong  buffers carry x, v, and a per-particle initial-stream tag).
6. `k_deposit`: per-node CIC charge deposition, gather form (no atomics).
7. `k_field`: cumulative integral of `(rho - <rho>)` → E, then subtract `<E>`.
8. `k_push`: leapfrog kick-drift, periodic wrap.

Normalization: `omega_p = 1`, `n_0 = 1`, `eps_0 = 1`, `m_e = 1`, `q_e = -1`. FP32 throughout (my M1 mac can't do FP64 lmao)

## Editor / LSP

`compile_flags.txt` is the clangd configuration. It lists the `src/` and `src/OpenCL/include` paths plus the `-isystem` ImGui paths so clangd resolves the same headers as the compiler without flagging ImGui's internals. It should 'just work' no matter which editor you are using, as long as it supports some form of clangd language server integration.
