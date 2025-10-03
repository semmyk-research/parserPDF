# opted for sigleton as opposed to global variable

# Create a singleton object to hold all shared states
# This ensures that only one instance of the Config class is ever created
class Config:
    """ Single model_dict use across the app"""
    def __init__(self):
        self.model_dict = {}
        self.weasyprint_libpath = ""
        self.config_ini = "utils\\config.ini"
        self.pdf_files_count = 0

# Create a single, shared instance of the Config class
# Other modules will import and use this instance.
config_load_models = Config()

