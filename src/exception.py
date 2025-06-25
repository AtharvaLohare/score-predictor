# the sys module helps to get all the details about the exception that 
# has occured.

import sys
from src.logger import logging

def create_error_message(error, error_details:sys):
    """
    This function takes in the Exception object : "error" and 
    the sys module/package to return error message with more details
    as extracted from the sys.
    """
    _, _, exc_traceback = error_details.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    error_message="Error has occured in Python script named" \
                  "[{0}] in line number [{1}]." \
                  "\n Error message : [{2}].".format(
                      file_name,
                      exc_traceback.tb_lineno, 
                      str(error)
                  ) 
    return error_message


class CustomException(Exception):
    """ This is custom exception that provides more details."""
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = create_error_message(error_message, error_details)
    
    def __str__(self):
        return self.error_message