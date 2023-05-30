import os
import sys

from jinja2 import Environment, FileSystemLoader, select_autoescape

_default_skip_file_list = ["__init__.py"]
_default_skip_dir_list = ["__pycache__"]
_conflict_name_dict = {
            "manager": ["eviction", "object_data", "scalar_data", "vector_data"]
        }


class DocGen:
    def __init__(
        self,
        lib_name="gptcache",
        source_dir="../gptcache",
        output_dir="references",
        skip_list=[],
    ):
        self.lib_name = lib_name
        self.output_dir = os.path.abspath(output_dir)
        self.source_dir = os.path.abspath(source_dir)
        self.skip_list = skip_list

    @staticmethod
    def title_bar(input_str):
        return "=" * len(input_str)

    @staticmethod
    def section_bar(input_str):
        return "-" * len(input_str)

    @staticmethod
    def get_filename(input_str):
        if input_str == "gptcache":
            return input_str
        suffix = os.path.splitext(input_str)[1][1:]
        for conflict_dir, conflict_names in _conflict_name_dict.items():
            for conflict_name in conflict_names:
                if f"{conflict_dir}.{conflict_name}" in input_str:
                    return f"{conflict_name}.{suffix}"

        return suffix

    @staticmethod
    def cap(input_str):
        input_str = DocGen.get_filename(input_str)
        if input_str == "gptcache":
            return "GPTCache"
        return str.join(" ", [i.capitalize() for i in input_str.split("_")])

    def model_name(self, input_str: str):
        return self.lib_name + input_str[len(self.source_dir) :].replace("/", ".")

    def get_module_and_libs(self, module_dir, is_root):
        module = self.model_name(module_dir)
        libs = []
        for file in os.listdir(module_dir):
            if (
                os.path.isfile(os.path.join(module_dir, file))
                and file not in _default_skip_file_list
            ):
                libs.append(module + "." + os.path.splitext(file)[0])
            if not is_root:
                if (
                    os.path.isdir(os.path.join(module_dir, file))
                    and file not in _default_skip_dir_list
                ):
                    _, child_libs = self.get_module_and_libs(
                        os.path.join(module_dir, file), False
                    )
                    libs.extend(child_libs)
        if len(libs) > 0:
            sorted(libs)
            return module, libs
        return "", []

    def generate(self):
        # Set the output directory
        env = Environment(
            loader=FileSystemLoader(os.path.join(self.output_dir, "../_templates")),
            autoescape=select_autoescape(),
        )

        # Add custom filters
        env.filters["title_bar"] = DocGen.title_bar
        env.filters["section_bar"] = DocGen.section_bar
        env.filters["cap"] = DocGen.cap

        # Add the target path to the system path
        sys.path.insert(0, os.path.abspath(".."))

        # Load the modules
        modules = []
        libs = []

        a, b = self.get_module_and_libs(self.source_dir, True)
        if a:
            modules.append(a)
            libs.append(b)

        for file in os.listdir(self.source_dir):
            tmp_dir = os.path.join(self.source_dir, file)
            if os.path.isdir(tmp_dir) and file not in _default_skip_dir_list:
                a, b = self.get_module_and_libs(tmp_dir, False)
                if a:
                    modules.append(a)
                    libs.append(b)

        # Render the index templates and write rendered output to files
        index_temp = env.get_template("index.rst")

        with open(os.path.join(self.output_dir, "index.rst"), "w") as f:
            t = index_temp.render(
                {"modules": [DocGen.get_filename(module) for module in modules]}
            )
            f.write(t)

        # Render the function templates and write rendered output to files
        func_temp = env.get_template("function.rst")

        for index, module in enumerate(modules):
            with open(
                os.path.join(self.output_dir, f"{DocGen.get_filename(module)}.rst"), "w"
            ) as f:
                t = func_temp.render(
                    {
                        "module_name": module,
                        "funcs": [
                            (DocGen.get_filename(lib), lib) for lib in libs[index]
                        ],
                    }
                )
                f.write(t)


# if __name__ == "__main__":
#     gen = DocGen(source_dir="/Users/derek/fubang/gptcache/gptcache", output_dir="/Users/derek/fubang/gptcache/docs/references")
#     gen.generate()
