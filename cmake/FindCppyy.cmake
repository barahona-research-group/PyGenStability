#.rst:
# FindCppyy
# -------
#
# Find Cppyy
#
# This module finds an installed Cppyy.  It sets the following variables:
#
# ::
#
#   Cppyy_FOUND - set to true if Cppyy is found
#   Cppyy_DIR - the directory where Cppyy is installed
#   Cppyy_EXECUTABLE - the path to the cppyy-generator executable
#   Cppyy_INCLUDE_DIRS - Where to find the ROOT header files.
#   Cppyy_VERSION - the version number of the Cppyy backend.
#
#
# The module also defines the following functions:
#
#   cppyy_add_bindings - Generate a set of bindings from a set of header files.
#
# The minimum required version of Cppyy can be specified using the
# standard syntax, e.g.  find_package(Cppyy 4.19)
#
#

execute_process(COMMAND cling-config --cmake OUTPUT_VARIABLE CPPYY_MODULE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

if(CPPYY_MODULE_PATH)
    #
    # Cppyy_DIR: one level above the installed cppyy cmake module path
    #
    set(Cppyy_DIR ${CPPYY_MODULE_PATH}/../)
    #
    # Cppyy_INCLUDE_DIRS: Directory with cppyy headers
    #
    set(Cppyy_INCLUDE_DIRS ${Cppyy_DIR}include)
    #
    # Cppyy_VERSION.
    #
    find_package(ROOT QUIET REQUIRED PATHS ${CPPYY_MODULE_PATH})
    if(ROOT_FOUND)
        set(Cppyy_VERSION ${ROOT_VERSION})
    endif()
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cppyy
                                  REQUIRED_VARS ROOT_genreflex_CMD Cppyy_DIR Cppyy_INCLUDE_DIRS CPPYY_MODULE_PATH
                                  VERSION_VAR Cppyy_VERSION
)
mark_as_advanced(Cppyy_VERSION)

# Get the cppyy libCling library. Not sure if necessary?
find_library(LibCling_LIBRARY libCling.so PATHS ${Cppyy_DIR}/lib)


#
# Generate setup.py from the setup.py.in template.
#
function(cppyy_generate_setup pkg version lib_so_file rootmap_file pcm_file map_file)
    set(SETUP_PY_FILE ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
    set(CPPYY_PKG ${pkg})
    get_filename_component(CPPYY_LIB_SO ${lib_so_file} NAME)
    get_filename_component(CPPYY_ROOTMAP ${rootmap_file} NAME)
    get_filename_component(CPPYY_PCM ${pcm_file} NAME)
    get_filename_component(CPPYY_MAP ${map_file} NAME)
    configure_file(${CMAKE_SOURCE_DIR}/pkg_templates/setup.py.in ${SETUP_PY_FILE})

    set(SETUP_PY_FILE ${SETUP_PY_FILE} PARENT_SCOPE)
endfunction(cppyy_generate_setup)

#
# Generate a packages __init__.py using the __init__.py.in template.
#
function(cppyy_generate_init)
    set(simple_args PKG LIB_FILE MAP_FILE)
    set(list_args NAMESPACES)
    cmake_parse_arguments(ARG
                          ""
                          "${simple_args}"
                          "${list_args}"
                          ${ARGN}
    )

    set(INIT_PY_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_PKG}/__init__.py)
    set(CPPYY_PKG ${ARG_PKG})
    get_filename_component(CPPYY_LIB_SO ${ARG_LIB_FILE} NAME)
    get_filename_component(CPPYY_MAP ${ARG_MAP_FILE} NAME)

    list(JOIN ARG_NAMESPACES ", " _namespaces)

    if(NOT "${ARG_NAMESPACES}" STREQUAL "")
        list(JOIN ARG_NAMESPACES ", " _namespaces)
        set(NAMESPACE_INJECTIONS "from cppyy.gbl import ${_namespaces}")
    else()
        set(NAMESPACE_INJECTIONS "")
    endif()

    configure_file(${CMAKE_SOURCE_DIR}/pkg_templates/__init__.py.in ${INIT_PY_FILE})

    set(INIT_PY_FILE ${INIT_PY_FILE} PARENT_SCOPE)
endfunction(cppyy_generate_init)

#
# Generate a set of bindings from a set of header files. Somewhat like CMake's
# add_library(), the output is a compiler target. In addition ancilliary files
# are also generated to allow a complete set of bindings to be compiled,
# packaged and installed.
#
#   cppyy_add_bindings(
#       pkg
#       pkg_version
#       author
#       author_email
#       [URL url]
#       [LICENSE license]
#       [LANGUAGE_STANDARD std]
#       [GENERATE_OPTIONS option...]
#       [COMPILE_OPTIONS option...]
#       [INCLUDE_DIRS dir...]
#       [LINK_LIBRARIES library...]
#
# The bindings are based on https://cppyy.readthedocs.io/en/latest/, and can be
# used as per the documentation provided via the cppyy.gbl namespace. First add
# the directory of the <pkg>.rootmap file to the LD_LIBRARY_PATH environment
# variable, then "import cppyy; from cppyy.gbl import <some-C++-entity>".
#
# Alternatively, use "import <pkg>". This convenience wrapper supports
# "discovery" of the available C++ entities using, for example Python 3's command
# line completion support.
#
# This function creates setup.py, setup.cfg, and MANIFEST.in appropriate
# for the package in the build directory. It also creates the package directory PKG,
# and within it a tests subdmodule PKG/tests/test_bindings.py to sanity test the bindings.
# Further, it creates PKG/pythonizors/, which can contain files of the form
# pythonize_*.py, with functions of the form pythonize_<NAMESPACE>_*.py, which will
# be consumed by the initialization routine and added as pythonizors for their associated
# namespace on import.
# 
# The setup.py and setup.cfg are prepared to create a Wheel. They can be customized
# for the particular package by modifying the templates in pkg_templates/.
#
# The bindings are generated/built/packaged using 3 environments:
#
#   - One compatible with the header files being bound. This is used to
#     generate the generic C++ binding code (and some ancilliary files) using
#     a modified C++ compiler. The needed options must be compatible with the
#     normal build environment of the header files.
#
#   - One to compile the generated, generic C++ binding code using a standard
#     C++ compiler. The resulting library code is "universal" in that it is
#     compatible with both Python2 and Python3.
#
#   - One to package the library and ancilliary files into standard Python2/3
#     wheel format. The packaging is done using native Python tooling.
#
# Arguments and options:
#
#   pkg                 The name of the package to generate. This can be either
#                       of the form "simplename" (e.g. "Akonadi"), or of the
#                       form "namespace.simplename" (e.g. "KF5.Akonadi").
#
#   pkg_version         The version of the package.
#
#   author              The name of the bindings author.
#
#   author_email        The email address of the bindings author.
#
#   URL url             The home page for the library or bindings. Default is
#                       "https://pypi.python.org/pypi/<pkg>".
#
#   LICENSE license     The license, default is "MIT".
#
#   LICENSE_FILE        Path to license file to include in package. Default is LICENSE.
#
#   README              Path to README file to include in package and use as
#                       text for long_description. Default is README.rst.
#
#   LANGUAGE_STANDARD std
#                       The version of C++ in use, "14" by default.
#
#   INTERFACE_FILE      Header to be passed to genreflex. Should contain template
#                       specialization declarations if required.
#
#   HEADERS             Library headers from which to generate the map. Should match up with
#                       interface file includes.
#
#   SELECTION_XML       selection XML file passed to genreflex.
#
#   
#
#   GENERATE_OPTIONS option
#                       Options which will be passed to the rootcling invocation
#                       in the cppyy-generate utility. cppyy-generate is used to
#                       create the bindings map.
# 
#  
#
#
#   COMPILE_OPTIONS option
#                       Options which are to be passed into the compile/link
#                       command.
#
#   INCLUDE_DIRS dir    Include directories.
#
#   LINK_LIBRARIES library
#                       Libraries to link against.
#
#   NAMESPACES          List of C++ namespaces which should be imported into the
#                       bindings' __init__.py. This avoids having to write imports
#                       of the form `from PKG import NAMESPACE`.
#
#   EXTRA_PKG_FILES     Extra files to copy into the package. Note that non-python
#                       files will need to be added to the MANIFEST.in.in template.
#
#
# Returns via PARENT_SCOPE variables:
#
#   CPPYY_LIB_TARGET    The target cppyy bindings shared library.
#
#   SETUP_PY_FILE       The generated setup.py.
#
function(cppyy_add_bindings pkg pkg_version author author_email)
    set(simple_args URL LICENSE LICENSE_FILE LANGUAGE_STANDARD INTERFACE_FILE
        SELECTION_XML README_FILE)
    set(list_args HEADERS  COMPILE_OPTIONS INCLUDE_DIRS LINK_LIBRARIES 
        GENERATE_OPTIONS NAMESPACES EXTRA_PKG_FILES)
    cmake_parse_arguments(
        ARG
        ""
        "${simple_args}"
        "${list_args}"
        ${ARGN})
    if(NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(SEND_ERROR "Unexpected arguments specified '${ARG_UNPARSED_ARGUMENTS}'")
    endif()
    string(REGEX MATCH "[^\.]+$" pkg_simplename ${pkg})
    string(REGEX REPLACE "\.?${pkg_simplename}" "" pkg_namespace ${pkg})
    set(pkg_dir ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "." "/" tmp ${pkg})
    set(pkg_dir "${pkg_dir}/${tmp}")
    set(lib_name "${pkg_namespace}${pkg_simplename}Cppyy")
    set(lib_file ${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(cpp_file ${CMAKE_CURRENT_BINARY_DIR}/${pkg_simplename}.cpp)
    set(pcm_file ${pkg_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}_rdict.pcm)
    set(rootmap_file ${pkg_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib_name}.rootmap)
    set(extra_map_file ${pkg_dir}/${pkg_simplename}.map)

    #
    # Package metadata.
    #
    if("${ARG_URL}" STREQUAL "")
        string(REPLACE "." "-" tmp ${pkg})
        set(ARG_URL "https://pypi.python.org/pypi/${tmp}")
    endif()
    if("${ARG_LICENSE}" STREQUAL "")
        set(ARG_LICENSE "MIT")
    endif()
    set(BINDINGS_LICENSE ${ARG_LICENSE})

    if("${ARG_LICENSE_FILE}" STREQUAL "")
        set(ARG_LICENSE_FILE ${CMAKE_SOURCE_DIR}/LICENSE)
    endif()
    set(LICENSE_FILE ${ARG_LICENSE_FILE})

    if("${ARG_README_FILE}" STREQUAL "")
        set(ARG_README_FILE ${CMAKE_SOURCE_DIR}/README.rst)
    endif()
    set(README_FILE ${ARG_README_FILE})

    #
    # Language standard.
    #
    if("${ARG_LANGUAGE_STANDARD}" STREQUAL "")
        set(ARG_LANGUAGE_STANDARD "14")
    endif()

    #
    # Includes
    #
    foreach(dir ${ARG_INCLUDE_DIRS})
        list(APPEND includes "-I${dir}")
    endforeach()
    
    find_package(LibClang REQUIRED)

    #
    # Set up genreflex args.
    #
    set(genreflex_args)
    if("${ARG_INTERFACE_FILE}" STREQUAL "")
        message(SEND_ERROR "No Interface specified")
    endif()
    list(APPEND genreflex_args "${ARG_INTERFACE_FILE}")
    if(NOT "${ARG_SELECTION_XML}" STREQUAL "")
        list(APPEND genreflex_args "--selection=${ARG_SELECTION_XML}")
    endif()

    list(APPEND genreflex_args "-o" "${cpp_file}")
    list(APPEND genreflex_args "--rootmap=${rootmap_file}")
    list(APPEND genreflex_args "--rootmap-lib=${lib_file}")
    list(APPEND genreflex_args "-l" "${lib_file}")
    foreach(dir ${includes})
        list(APPEND genreflex_args "${dir}")
    endforeach(dir)

    set(genreflex_cxxflags "--cxxflags")
    list(APPEND genreflex_cxxflags "-std=c++${ARG_LANGUAGE_STANDARD}")

    #
    # run genreflex
    #
    file(MAKE_DIRECTORY ${pkg_dir})
    add_custom_command(OUTPUT ${cpp_file} ${rootmap_file} ${pcm_file}
                       COMMAND ${ROOT_genreflex_CMD} ${genreflex_args} ${genreflex_cxxflags}
                       DEPENDS ${ARG_INTERFACE_FILE}
                       WORKING_DIRECTORY ${pkg_dir}
    )

    #
    # Set up cppyy-generator args.
    #
    list(APPEND ARG_GENERATE_OPTIONS "-std=c++${ARG_LANGUAGE_STANDARD}")
    if(${CONDA_ACTIVE})
        set(CLANGDEV_INCLUDE $ENV{CONDA_PREFIX}/lib/clang/${CLANG_VERSION_STRING}/include)
        message(STATUS "adding conda clangdev includes to cppyy-generator options (${CLANGDEV_INCLUDE})")
        list(APPEND ARG_GENERATE_OPTIONS "-I${CLANGDEV_INCLUDE}")
    endif()
    #
    # Run cppyy-generator. First check dependencies. TODO: temporary hack: rather
    # than an external dependency, enable libclang in the local build.
    #
    get_filename_component(Cppyygen_EXECUTABLE ${ROOT_genreflex_CMD} DIRECTORY)
    set(Cppyygen_EXECUTABLE ${Cppyygen_EXECUTABLE}/cppyy-generator)

    set(generator_args)
    foreach(arg IN LISTS ARG_GENERATE_OPTIONS)
        string(REGEX REPLACE "^-" "\\\\-" arg ${arg})
        list(APPEND generator_args ${arg})
    endforeach()

    add_custom_command(OUTPUT ${extra_map_file}
                       COMMAND ${LibClang_PYTHON_EXECUTABLE} ${Cppyygen_EXECUTABLE} 
                               --libclang ${LibClang_LIBRARY} --flags "\"${generator_args}\""
                               ${extra_map_file} ${ARG_HEADERS}
                       DEPENDS ${ARG_HEADERS} 
                       WORKING_DIRECTORY ${pkg_dir}
    )
    #
    # Compile/link.
    #
    add_library(${lib_name} SHARED ${cpp_file} ${pcm_file} ${rootmap_file} ${extra_map_file})
    set_target_properties(${lib_name} PROPERTIES LINKER_LANGUAGE CXX)
    set_property(TARGET ${lib_name} PROPERTY VERSION ${version})
    set_property(TARGET ${lib_name} PROPERTY CXX_STANDARD ${ARG_LANGUAGE_STANDARD})
    set_property(TARGET ${lib_name} PROPERTY LIBRARY_OUTPUT_DIRECTORY ${pkg_dir})
    set_property(TARGET ${lib_name} PROPERTY LINK_WHAT_YOU_USE TRUE)
    target_include_directories(${lib_name} PRIVATE ${Cppyy_INCLUDE_DIRS} ${ARG_INCLUDE_DIRS})
    target_compile_options(${lib_name} PRIVATE ${ARG_COMPILE_OPTIONS})
    target_link_libraries(${lib_name} PUBLIC ${LibCling_LIBRARY} ${ARG_LINK_LIBRARIES})

    #
    # Generate __init__.py
    #
    cppyy_generate_init(PKG        ${pkg}
                        LIB_FILE   ${lib_file}
                        MAP_FILE   ${extra_map_file}
                        NAMESPACES ${ARG_NAMESPACES}
    )
    set(INIT_PY_FILE ${INIT_PY_FILE} PARENT_SCOPE)

    #
    # Generate setup.py
    #
    cppyy_generate_setup(${pkg}
                         ${pkg_version}
                         ${lib_file}
                         ${rootmap_file}
                         ${pcm_file}
                         ${extra_map_file}
    )

    #
    # Generate setup.cfg
    #
    set(setup_cfg ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg)
    configure_file(${CMAKE_SOURCE_DIR}/pkg_templates/setup.cfg.in ${setup_cfg})

    #
    # Copy README and LICENSE
    #
    file(COPY ${README_FILE}  DESTINATION . USE_SOURCE_PERMISSIONS)
    file(COPY ${LICENSE_FILE} DESTINATION . USE_SOURCE_PERMISSIONS)

    #
    # Generate a pytest/nosetest sanity test script.
    #
    set(PKG ${pkg})
    configure_file(${CMAKE_SOURCE_DIR}/pkg_templates/test_bindings.py.in ${pkg_dir}/tests/test_bindings.py)

    #
    # Generate MANIFEST.in 
    #
    configure_file(${CMAKE_SOURCE_DIR}/pkg_templates/MANIFEST.in.in ${CMAKE_CURRENT_BINARY_DIR}/MANIFEST.in)

    #
    # Copy pure python code
    #
    file(COPY ${CMAKE_SOURCE_DIR}/py/ DESTINATION ${pkg_dir}
         USE_SOURCE_PERMISSIONS
         FILES_MATCHING PATTERN "*.py")

    #
    # Copy any extra files into package.
    #
    file(COPY ${ARG_EXTRA_FILES} DESTINATION ${pkg_dir} USE_SOURCE_PERMISSIONS)

    #
    # Kinda ugly: you'e not really supposed to glob like this. Oh well. Using this to set
    # dependencies for the python wheel building command; the file copy above is done on every
    # cmake invocation anyhow.
    #
    # Then, get the system architecture and build the wheel string based on PEP 427.
    #
    file(GLOB_RECURSE PY_PKG_FILES
         LIST_DIRECTORIES FALSE
         CONFIGURE_DEPENDS
         "${CMAKE_SOURCE_DIR}/py/*.py")
    string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_STR)
    set(pkg_whl "${CMAKE_BINARY_DIR}/dist/${pkg}-${pkg_version}-py3-none-${SYSTEM_STR}_${CMAKE_SYSTEM_PROCESSOR}.whl")
    add_custom_command(OUTPUT  ${pkg_whl}
                       COMMAND ${LibClang_PYTHON_EXECUTABLE} setup.py bdist_wheel
                       DEPENDS ${SETUP_PY_FILE} ${lib_name} ${setup_cfg}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    add_custom_target(wheel ALL
                      DEPENDS ${pkg_whl}
    )
    add_dependencies(wheel ${lib_name})

    #
    # Return results.
    #
    set(LIBCLING         ${LibCling_LIBRARY} PARENT_SCOPE)
    set(CPPYY_LIB_TARGET ${lib_name} PARENT_SCOPE)
    set(SETUP_PY_FILE    ${SETUP_PY_FILE} PARENT_SCOPE)
    set(PY_WHEEL_FILE    ${pkg_whl}  PARENT_SCOPE)
endfunction(cppyy_add_bindings)


#
# Return a list of available pip programs.
#
function(cppyy_find_pips)
    execute_process(
        COMMAND python -c "from cppyy_backend import bindings_utils; print(\";\".join(bindings_utils.find_pips()))"
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        RESULT_VARIABLE _rc
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${_rc}" STREQUAL "0")
        message(FATAL_ERROR "Error finding pips: (${_rc}) ${_stderr}")
    endif()
    set(PIP_EXECUTABLES ${_stdout} PARENT_SCOPE)
endfunction(cppyy_find_pips)
