import inspect

def get_arg_name_as_string(arg):
    """
    Returns the name of the argument passed to the function as a string.
    This works by inspecting the calling frame's local variables.

    example usage:
    def my_function(x):
        arg_name = get_arg_name_as_string(arg_x)
        print(f"The argument name is: {arg_name}")  # Outputs: "The argument name is: arg_x"
    """
    frame = inspect.currentframe().f_back # Get the frame of the caller
    arg_name = None
    for name, value in frame.f_locals.items():
        if value is arg:
            arg_name = name
            break
    return arg_name
