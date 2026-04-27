# Build for the two-stream electrostatic PIC demo.
#
# Targets:
#   make             — build bin/two_stream
#   make run         — build and run (forward args via ARGS, e.g. `make run ARGS=0`)
#   make clean       — remove app objects and binary
#   make distclean   — clean plus remove the ImGui object cache and static archive
#   make rebuild-imgui — force-rebuild the ImGui static archive
#
# System dependencies (one-time):
#   macOS: brew install glfw
#   Linux: sudo apt install libglfw3-dev libglew-dev ocl-icd-libopencl1 ocl-icd-opencl-dev
#
# Dear ImGui is auto-fetched into extern/imgui on first build, then compiled
# once into extern/imgui/build/libimgui.a and reused on subsequent builds.

CXX        ?= g++
CXXSTD     ?= -std=c++17
OPT        ?= -O3
APP_WARN   := -Wno-comment -Wno-unused-result -Wno-deprecated-declarations

# ---- Layout ---------------------------------------------------------------
BIN_DIR    := bin
OBJ_DIR    := build
IMGUI_DIR  := extern/imgui
IMGUI_BLD  := $(IMGUI_DIR)/build
IMGUI_LIB  := $(IMGUI_BLD)/libimgui.a
IMGUI_TAG  := v1.91.0
IMGUI_URL  := https://github.com/ocornut/imgui.git
TARGET     := $(BIN_DIR)/two_stream

# ---- Source / object lists -----------------------------------------------
APP_SRCS   := two_stream.cpp src/kernel.cpp
APP_OBJS   := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(APP_SRCS))
APP_DEPS   := $(APP_OBJS:.o=.d)

IMGUI_SRCS := \
    $(IMGUI_DIR)/imgui.cpp \
    $(IMGUI_DIR)/imgui_draw.cpp \
    $(IMGUI_DIR)/imgui_tables.cpp \
    $(IMGUI_DIR)/imgui_widgets.cpp \
    $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
    $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp,$(IMGUI_BLD)/%.o,$(IMGUI_SRCS))

# ---- Include / library flags ---------------------------------------------
# Self-contained: opencl.hpp, utilities.hpp, and the OpenCL Khronos headers
# all live under src/ — no dependency on a sibling project tree.
SRC_INC       := -Isrc -Isrc/OpenCL/include
# Use -I when ImGui builds itself (so its private headers resolve normally),
# -isystem elsewhere so ImGui's internal warnings don't bleed into our compile.
IMGUI_INC     := -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
IMGUI_INC_SYS := -isystem $(IMGUI_DIR) -isystem $(IMGUI_DIR)/backends

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  BREW_PREFIX  := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
  PLATFORM_INC := -I$(BREW_PREFIX)/include
  PLATFORM_LIB := -L$(BREW_PREFIX)/lib
  PLATFORM_LIBS:= -lglfw -framework OpenCL -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
else ifeq ($(UNAME_S),Linux)
  PLATFORM_INC :=
  PLATFORM_LIB :=
  PLATFORM_LIBS:= -lglfw -lGL -lGLEW -lOpenCL
else
  $(error Unsupported OS: $(UNAME_S))
endif

APP_CXXFLAGS   := $(CXXSTD) $(OPT) -pthread $(APP_WARN) $(SRC_INC) $(IMGUI_INC_SYS) $(PLATFORM_INC)
IMGUI_CXXFLAGS := $(CXXSTD) $(OPT) -w $(IMGUI_INC) $(PLATFORM_INC)

# ---- Phony targets --------------------------------------------------------
.DEFAULT_GOAL := all
.PHONY: all run clean distclean rebuild-imgui

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

distclean: clean
	rm -rf $(IMGUI_BLD)

rebuild-imgui:
	rm -rf $(IMGUI_BLD)
	$(MAKE) $(IMGUI_LIB)

# ---- Auto-fetch ImGui -----------------------------------------------------
# If imgui.cpp is missing, clone the repo. Make will rerun this only if the
# file disappears (e.g. user deleted extern/imgui/ or hasn't fetched yet).
$(IMGUI_DIR)/imgui.cpp:
	@echo "Fetching Dear ImGui $(IMGUI_TAG)..."
	@mkdir -p $(dir $(IMGUI_DIR))
	git clone --depth 1 --branch $(IMGUI_TAG) $(IMGUI_URL) $(IMGUI_DIR)

# ---- ImGui object files + static archive ----------------------------------
# Order-only prereq on imgui.cpp guarantees the clone has run before any
# ImGui source is compiled, without re-triggering on its mtime.
$(IMGUI_BLD)/%.o: $(IMGUI_DIR)/%.cpp | $(IMGUI_DIR)/imgui.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(IMGUI_CXXFLAGS)

$(IMGUI_LIB): $(IMGUI_OBJS)
	@mkdir -p $(dir $@)
	@rm -f $@
	ar rcs $@ $^

# ---- Application object files (with auto header-deps) ---------------------
$(OBJ_DIR)/%.o: %.cpp | $(IMGUI_DIR)/imgui.cpp
	@mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ $(APP_CXXFLAGS) -MMD -MP

-include $(APP_DEPS)

# ---- Link -----------------------------------------------------------------
$(TARGET): $(APP_OBJS) $(IMGUI_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(APP_OBJS) $(IMGUI_LIB) -o $@ $(PLATFORM_LIB) $(PLATFORM_LIBS)
