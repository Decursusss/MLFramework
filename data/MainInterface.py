import os
import importlib.util
import inspect

class DataInterface:
    def __init__(self):
        self.SOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), "sources")

    def list_sources(self):
        list_sources = os.listdir(self.SOURCE_DIRECTORY)
        return list_sources

    def load_module(self, name: str):
        py_file = os.path.join(self.SOURCE_DIRECTORY, name, f"{name}.py")

        if not os.path.exists(py_file):
            raise FileNotFoundError(
                f"File not found: {py_file}\n"
                f"Available data sources: {self.list_sources()}"
            )

        spec = importlib.util.spec_from_file_location(name, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def source_functions(self, name:str):
        module = self.load_module(name)
        result = {}
        for func_name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__ and not func_name.startswith("_"):
                result[func_name] = inspect.signature(obj)
        return result

    def use_source_function(self, name: str, function: str, *args, **kwargs):
        module = self.load_module(name)

        if not hasattr(module, function):
            available = list(self.source_functions(name).keys())
            raise AttributeError(
                f"Function '{function}' not found in '{name}'\n"
                f"Available functions: {available}"
            )

        func = getattr(module, function)
        return func(*args, **kwargs)

    def info(self):
        sources = self.list_sources()
        if not sources:
            print("Not found any sources")
            return

        for index, source in enumerate(sources):
            print(f"\n {index}: {source}")
            try:
                funcs = self.source_functions(source)
                for func_name, sig in funcs.items():
                    print(f"   • {func_name}{sig}")
            except Exception as e:
                print(f"   Some Error Appeared: {e}")
