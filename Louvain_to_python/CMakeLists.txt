cmake_minimum_required(VERSION 3.12)

project(Louvain_to_python
        LANGUAGES CXX 
        VERSION 0.9
        HOMEPAGE_URL "https://github.com/ImperialCollegeLondon/PyGenStability"
	DESCRIPTION "Markov Stability based on Generalise louvain cpp code")

if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Building in a conda environment.")
    set(CONDA_ACTIVE TRUE)
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    #include_directories($ENV{CONDA_PREFIX}/include)
    set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
endif()

set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# headers are listed in a cmake file.
include(${CMAKE_SOURCE_DIR}/manifest.cmake)
include(GNUInstallDirs)

#
# Add our project's cmake dir the the module path. This gives us the
# Cppyy commands and targets.
#
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(Cppyy)






#
# Make the default build us c++14 and "RELEASE" (-O3)
#
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads)


find_package(Lemon REQUIRED)


#find_package(Lemon REQUIRED)
#include_directories(${LEMON_INCLUDE_DIRS})
#set(LIBS ${LIBS} ${LEMON_LIBRARIES})


#
# Make an object file with the cpp header 
#

add_library(louvain_to_python SHARED
            ${LIB_HEADERS}
	    ${LEMON_LIBS} 
)



set_target_properties(louvain_to_python PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(louvain_to_python PROPERTIES PUBLIC_HEADER ${LIB_HEADERS})
target_include_directories(louvain_to_python
                           PUBLIC
                           ${CMAKE_SOURCE_DIR}
			   ${LEMON_INCLUDE_DIRS}
)
target_link_libraries(louvain_to_python
                      PRIVATE
                      Threads::Threads
)


#
# Set up the Cppyy bindings generation. This is a customized version defined
# in boink's cmake/ dir; it uses genreflex rather than calling rootcling directly.
# I did this because I couldn't get rootcling to properly include/exclude classes
# via the LinkDef header, and I wanted to be able to use the better syntax in
# the genreflex selection XML anyhow. Also, I think this is now the recommended /
# more modern way anyhow? Code was modified from the versions cppyy distributes.
#
cppyy_add_bindings(
	"Louvain_to_python" "${PROJECT_VERSION}" "Alexis Arnaudon" "alexis.arnaudon@imperial.ac.uk"
   LICENSE "MIT"
   LANGUAGE_STANDARD "14"
   SELECTION_XML  ${CMAKE_SOURCE_DIR}/selection.xml
   INTERFACE_FILE ${CMAKE_SOURCE_DIR}/interface.hh
   HEADERS        ${LIB_HEADERS} ${CMAKE_SOURCE_DIR}/interface.hh
   INCLUDE_DIRS   ${CMAKE_SOURCE_DIR} ${LEMON_INCLUDE_DIR}
   LINK_LIBRARIES louvain_to_python
   NAMESPACES     clq 
)


install(TARGETS       louvain_to_python
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/louvain_to_python
)

install(CODE "execute_process(COMMAND pip install ${PY_WHEEL_FILE})")
