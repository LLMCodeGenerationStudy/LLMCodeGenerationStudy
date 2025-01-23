from importlib import import_module

def dynamic_import_function(function_path):
    '''Dynamically import a function from a path string by using import_module() (e.g., "module.submodule.my_function")
    if no match distribution found, please return ModuleNotFoundError
    Args:
        a function path string:string
    Returns:
        function (e.g., my_function)
    '''
    # ----
    
    try:
        module_path, function_name = function_path.rsplit(".", 1)
        module = import_module(module_path)
        function = getattr(module, function_name)
    except:
        return ModuleNotFoundError
    return function


# unit test cases
print(dynamic_import_function('math.sqrt'))
print(dynamic_import_function('os.path.join'))
print(dynamic_import_function('nonexistent.mod.func'))