import os
import sys
import inspect
from jinja2 import Environment, FileSystemLoader, select_autoescape

class DocGen:

    def __init__(self, output_dir='references', skip_list=[]):
        _default_skip_list = ['__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__']

        self.OUTPUT = os.path.abspath(output_dir)
        self.skip_list = _default_skip_list + skip_list
    
    @staticmethod
    def title_bar(input):
        return "="*len(input)

    @staticmethod
    def section_bar(input):
        return "-"*len(input)

    @staticmethod
    def cap(input):
        r = ''
        if input == 'gptcache':
            r = 'GPTCache'
        else:
            r = str.capitalize(input)

        return r
    
    def generate(self, lib_name):
        # Set the output directory
        env = Environment(
            loader=FileSystemLoader("_templates"),
            autoescape=select_autoescape()
        )

        # Add custom filters
        env.filters['title_bar'] = DocGen.title_bar
        env.filters['section_bar'] = DocGen.section_bar
        env.filters['cap'] = DocGen.cap

        # Add the target path to the system path
        sys.path.insert(0, os.path.abspath('..')) 

        # Import the library
        try:
            lib = __import__(lib_name)
        except ImportError:
            print(f"Can't import {lib_name}")
            return

        # Get the modules, functions, and classes
        modules = [ x for x in inspect.getmembers(lib) if inspect.ismodule(x[1]) and str.startswith(x[1].__name__, f'{lib_name}.') ]
        functions = [ x for x in inspect.getmembers(lib) if inspect.isfunction(x[1]) ]
        classes = [ x for x in inspect.getmembers(lib) if inspect.isclass(x[1]) ]

        cf_combined = classes + functions
        cf_combined = [ x + (f"{lib_name}.{x[0]}",) for x in cf_combined if x[0] not in self.skip_list ]

        # Render the index templates and write rendered output to files
        index_temp = env.get_template("index.rst")

        with open(os.path.join(self.OUTPUT, "index.rst"), 'w') as f:
            t = index_temp.render({
                "modules": [(lib_name, "")] + modules
            })
            f.write(t)

        # Render the function templates and write rendered output to files
        func_temp = env.get_template("function.rst")

        with open(os.path.join(self.OUTPUT, f"{lib_name}.rst"), 'w') as f:
            t = func_temp.render({
                "module_name": lib_name,
                "funcs": cf_combined
            })

            f.write(t)
        
        # Iterate the modules, render the function templates and write rendered output to files
        for module in modules:
            module_name = module[0]
            ms = [ x for x in inspect.getmembers(module[1]) if inspect.ismodule(x[1]) ]
            ms = [ x + (f"{lib_name}.{module_name}.{x[0]}",) for x in ms if x[0] not in self.skip_list ]
            t = func_temp.render({
                "module_name": module[0],
                "funcs": ms
            })

            with open(os.path.join(self.OUTPUT, f"{module_name}.rst"), 'w') as f:
                f.write(t)


if __name__ == '__main__':
    gen = DocGen()
    gen.generate('gptcache')