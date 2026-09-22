if(NOT IS_ABSOLUTE "${DESTINATION}" OR EXISTS "${DESTINATION}" OR IS_SYMLINK "${DESTINATION}")
    message(FATAL_ERROR "Bundle destination must be an absolute new path; existing data is never deleted")
endif()
