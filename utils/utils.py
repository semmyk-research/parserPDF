
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


def get_time_now_str(tz_hours=None, date_format:str='%Y-%m-%d'):  #date_format:str='%d%b%Y'):
    """Returns the current time in a specific format + local time: ("%Y-%m-%d %H:%M:%S.%f %Z")."""
    from datetime import datetime, timezone, timedelta

    # Get the current time or UTC time
    if tz_hours is not None:
        current_utc_time = datetime.now(tz=timezone.utc) + timedelta(hours=tz_hours)
        current_time = current_utc_time
    else:
        current_time = datetime.now()

    # Format the time as a string
    #formatted_time = current_utc_time.strftime(date_format)  #("%Y-%m-%d %H:%M:%S.%f %Z")
    formatted_time = current_time.strftime(date_format)  #("%Y-%m-%d %H:%M:%S.%f %Z")

    #print(f"Current time: {formatted_time}")   ##debug
    return formatted_time

#get_time_now_str() ##debug

