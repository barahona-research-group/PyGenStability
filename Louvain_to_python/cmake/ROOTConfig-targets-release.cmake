#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ROOT::Cling" for configuration "Release"
set_property(TARGET ROOT::Cling APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::Cling PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libCling.so"
  IMPORTED_SONAME_RELEASE "libCling.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::Cling )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::Cling "${_IMPORT_PREFIX}/lib/libCling.so" )

# Import target "ROOT::Thread" for configuration "Release"
set_property(TARGET ROOT::Thread APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::Thread PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libThread.so"
  IMPORTED_SONAME_RELEASE "libThread.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::Thread )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::Thread "${_IMPORT_PREFIX}/lib/libThread.so" )

# Import target "ROOT::Core" for configuration "Release"
set_property(TARGET ROOT::Core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::Core PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libCore.so"
  IMPORTED_SONAME_RELEASE "libCore.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::Core )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::Core "${_IMPORT_PREFIX}/lib/libCore.so" )

# Import target "ROOT::rmkdepend" for configuration "Release"
set_property(TARGET ROOT::rmkdepend APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::rmkdepend PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/rmkdepend"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::rmkdepend )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::rmkdepend "${_IMPORT_PREFIX}/bin/rmkdepend" )

# Import target "ROOT::MathCore" for configuration "Release"
set_property(TARGET ROOT::MathCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::MathCore PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libMathCore.so"
  IMPORTED_SONAME_RELEASE "libMathCore.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::MathCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::MathCore "${_IMPORT_PREFIX}/lib/libMathCore.so" )

# Import target "ROOT::RIO" for configuration "Release"
set_property(TARGET ROOT::RIO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::RIO PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libRIO.so"
  IMPORTED_SONAME_RELEASE "libRIO.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::RIO )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::RIO "${_IMPORT_PREFIX}/lib/libRIO.so" )

# Import target "ROOT::rootcling" for configuration "Release"
set_property(TARGET ROOT::rootcling APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ROOT::rootcling PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/rootcling"
  )

list(APPEND _IMPORT_CHECK_TARGETS ROOT::rootcling )
list(APPEND _IMPORT_CHECK_FILES_FOR_ROOT::rootcling "${_IMPORT_PREFIX}/bin/rootcling" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
