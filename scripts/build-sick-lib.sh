#!/usr/bin/env bash

# Based off of: https://github.com/SICKAG/sick_scan_xd/blob/master/doc/sick_scan_api/sick_scan_api.md

FINAL_PROJECT_PATH="$(git rev-parse --show-toplevel)/FinalProject"
LIB_PATH="$FINAL_PROJECT_PATH/lib"
BUILD_PATH="$FINAL_PROJECT_PATH/build/SICK"

# Make sure library submodules are pulled
# git pull --recurse-submodules

mkdir -p $BUILD_PATH

# Build libsick_ldmrs
mkdir -p $LIB_PATH/libsick_ldmrs/build
cd $LIB_PATH/libsick_ldmrs/build
cmake -G "Unix Makefiles" ..
make -j4
sudo make -j4 install

mkdir -p $LIB_PATH/msgpack11/build
cd $LIB_PATH/msgpack11/build
cmake -DMSGPACK11_BUILD_TESTS=0 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -G "Unix Makefiles" ..
make -j4
sudo make -j4 install

cd $BUILD_PATH
export ROS_VERSION=0
cmake -DROS_VERSION=0 -G "Unix Makefiles" $LIB_PATH/sick_scan_xd
make -j4
sudo make -j4 install

ls -al ./sick_generic_caller
ls -al ./libsick_scan_xd_shared_lib.so
ls -al ./sick_scan_xd_api_test
ldd -r ./libsick_scan_xd_shared_lib.so
