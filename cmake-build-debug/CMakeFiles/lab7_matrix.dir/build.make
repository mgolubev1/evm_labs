# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/maxim/CLionProjects/lab7_matrix

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lab7_matrix.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab7_matrix.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab7_matrix.dir/flags.make

CMakeFiles/lab7_matrix.dir/main.cpp.o: CMakeFiles/lab7_matrix.dir/flags.make
CMakeFiles/lab7_matrix.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lab7_matrix.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lab7_matrix.dir/main.cpp.o -c /mnt/c/Users/maxim/CLionProjects/lab7_matrix/main.cpp

CMakeFiles/lab7_matrix.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lab7_matrix.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/maxim/CLionProjects/lab7_matrix/main.cpp > CMakeFiles/lab7_matrix.dir/main.cpp.i

CMakeFiles/lab7_matrix.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lab7_matrix.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/maxim/CLionProjects/lab7_matrix/main.cpp -o CMakeFiles/lab7_matrix.dir/main.cpp.s

# Object files for target lab7_matrix
lab7_matrix_OBJECTS = \
"CMakeFiles/lab7_matrix.dir/main.cpp.o"

# External object files for target lab7_matrix
lab7_matrix_EXTERNAL_OBJECTS =

lab7_matrix: CMakeFiles/lab7_matrix.dir/main.cpp.o
lab7_matrix: CMakeFiles/lab7_matrix.dir/build.make
lab7_matrix: CMakeFiles/lab7_matrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lab7_matrix"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab7_matrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab7_matrix.dir/build: lab7_matrix

.PHONY : CMakeFiles/lab7_matrix.dir/build

CMakeFiles/lab7_matrix.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab7_matrix.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab7_matrix.dir/clean

CMakeFiles/lab7_matrix.dir/depend:
	cd /mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/maxim/CLionProjects/lab7_matrix /mnt/c/Users/maxim/CLionProjects/lab7_matrix /mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug /mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug /mnt/c/Users/maxim/CLionProjects/lab7_matrix/cmake-build-debug/CMakeFiles/lab7_matrix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab7_matrix.dir/depend

