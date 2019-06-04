set(_headers
    louvain_to_python.h
)

set(_sources
)

foreach (path ${_headers})
    list(APPEND LIB_HEADERS ${CMAKE_SOURCE_DIR}/louvain_to_python/${path})
endforeach(path)

foreach (path ${_sources})
    list(APPEND LIB_SOURCES ${CMAKE_SOURCE_DIR}/louvain_to_python/${path})
endforeach(path)
