#!/bin/bash

BASEDIR=$(dirname "$0")

# Compiles the C++ libraries.
cmake ${BASEDIR}/CMakeLists.txt
make -C ${BASEDIR}

# Removes the temp files.
rm ${BASEDIR}/CMakeCache.txt
rm ${BASEDIR}/cmake_install.cmake
rm -r ${BASEDIR}/CMakeFiles
rm ${BASEDIR}/Makefile