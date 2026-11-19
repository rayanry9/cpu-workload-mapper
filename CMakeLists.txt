cmake_minimum_required(VERSION 3.20)

project("Workload Mapper" LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
    "Enable/Disable generation of compile_commands.json" FORCE)

include(ExternalProject)

ExternalProject_Add(hotspot_lib_build
	SOURCE_DIR "${CMAKE_SOURCE_DIR}/extern/HotSpot"
	BUILD_IN_SOURCE 1
	CONFIGURE_COMMAND ""
	BUILD_COMMAND make
	INSTALL_COMMAND ""
)

add_library(hotspot_lib STATIC IMPORTED GLOBAL)
set_target_properties(hotspot_lib PROPERTIES
	IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/extern/HotSpot/libhotspot.a"
	INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/extern/HotSpot"
)

add_dependencies(hotspot_lib hotspot_lib_build)

add_executable(workload_mapper src/main.c)
target_link_libraries(workload_mapper PRIVATE hotspot_lib m)
target_compile_definitions(workload_mapper PRIVATE 
	SOURCE_DIR="${CMAKE_SOURCE_DIR}")

