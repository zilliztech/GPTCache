import inspect
import os
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape


class DocGen:
    def __init__(self, output_dir="references", skip_list=[]):
        _default_skip_list = [
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        ]

        self.OUTPUT = os.path.abspath(output_dir)
        self.skip_list = _default_skip_list + skip_list
        self.expand_func_list = []
        self.add_list = [
            "adapter.api",
            "adapter.openai",
            "adapter.diffusers",
            "adapter.dolly",
            "adapter.langchain_models",
            "adapter.replicate",
            "adapter.stability_sdk",
        ]
        self.add_root_list = [
            "report",
            "session",
        ]

    @staticmethod
    def title_bar(input_str):
        return "=" * len(input_str)

    @staticmethod
    def section_bar(input_str):
        return "-" * len(input_str)

    @staticmethod
    def cap(input_str):
        return "GPTCache" if input_str == "gptcache" else str.join(" ", [i.capitalize() for i in input_str.split("_")])

    def generate(self, lib_name):
        # Set the output directory
        env = Environment(
            loader=FileSystemLoader("_templates"), autoescape=select_autoescape()
        )

        # Add custom filters
        env.filters["title_bar"] = DocGen.title_bar
        env.filters["section_bar"] = DocGen.section_bar
        env.filters["cap"] = DocGen.cap

        # Add the target path to the system path
        sys.path.insert(0, os.path.abspath(".."))

        # Import the library
        try:
            lib = __import__(lib_name)
        except ImportError:
            print(f"Can't import {lib_name}")
            return

        for x in self.add_list:
            sub_lib_name = lib_name + "." + x
            try:
                __import__(sub_lib_name)
            except ImportError:
                print(f"Can't import {sub_lib_name}")
                continue

        # Get the modules, functions, and classes
        modules = [
            x
            for x in inspect.getmembers(lib)
            if inspect.ismodule(x[1]) and str.startswith(x[1].__name__, f"{lib_name}.")
        ]
        functions = [x for x in inspect.getmembers(lib) if inspect.isfunction(x[1])]
        classes = [x for x in inspect.getmembers(lib) if inspect.isclass(x[1])]

        cf_combined = classes + functions
        classes_method = [
            (f"{y[0]}.{x[0]}", x) for y in classes for x in inspect.getmembers(y[1]) if
            inspect.isfunction(x[1]) and not x[0].startswith("_")
        ]
        cf_combined.extend(classes_method)
        cf_combined.sort(key=lambda x: x[0])
        cf_combined = [
            x + (f"{lib_name}.{x[0]}",)
            for x in cf_combined
            if x[0] not in self.skip_list
        ]

        # Render the index templates and write rendered output to files
        index_temp = env.get_template("index.rst")

        with open(os.path.join(self.OUTPUT, "index.rst"), "w") as f:
            t = index_temp.render({"modules": [(lib_name, "")] + modules})
            f.write(t)

        # Render the function templates and write rendered output to files
        func_temp = env.get_template("function.rst")

        with open(os.path.join(self.OUTPUT, f"{lib_name}.rst"), "w") as f:
            t = func_temp.render({"module_name": lib_name, "funcs": cf_combined})

            f.write(t)

        # Iterate the modules, render the function templates and write rendered output to files
        print("modules:", modules)
        for module in modules:
            module_name = module[0]

            if module_name in self.expand_func_list:
                ms_func = [
                    x for x in inspect.getmembers(module[1]) if inspect.isfunction(x[1])
                ]
                ms.extend(ms_func)
            elif module_name not in self.add_root_list:
                ms = [
                    x for x in inspect.getmembers(module[1]) if inspect.ismodule(x[1])
                ]
            else:
                ms = []

            classes = [
                x for x in inspect.getmembers(module[1]) if inspect.isclass(x[1]) and x[1].__module__.startswith(f"{lib_name}.{module_name}")
            ]
            ms.extend(classes)
            class_method = [
                (f"{y[0]}.{x[0]}", x) for y in classes for x in inspect.getmembers(y[1]) if inspect.isfunction(x[1]) and not x[0].startswith("_")
            ]
            ms.extend(class_method)

            ms = [
                x + (f"{lib_name}.{module_name}.{x[0]}",)
                for x in ms
                if x[0] not in self.skip_list
            ]
            t = func_temp.render({"module_name": module[0], "funcs": ms})

            with open(os.path.join(self.OUTPUT, f"{module_name}.rst"), "w") as f:
                f.write(t)


if __name__ == "__main__":
    gen = DocGen(skip_list=[
        "processor.context",
        "adapter.adapter",
    ])
    gen.generate("gptcache")
