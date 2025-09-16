def is_dict(variable):
    """Checks if a variable is a dict."""
    if isinstance(variable, dict):
        return True
        
    return False

def is_list_of_dicts(variable):
    """Checks if a variable is a list containing only dicts."""
    
    if isinstance(variable, list):
        # Return True only if the list is empty or all elements are dicts.
        return all(isinstance(item, dict) for item in variable)
        
    return False