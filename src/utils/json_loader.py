import os
import json


class JsonDataLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".json")]
        )

    def _load_file(self, filename: str) -> dict:
        path = os.path.join(self.folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. iterate through all json files in the folder and pass them as a list of strings
    def get_all_files_as_strings(self) -> list[str]:
        file_strings = []
        for filename in self.files:
            path = os.path.join(self.folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                file_strings.append(f.read())
        return file_strings

    # 2. return a specific file as a string by name
    def get_file_by_name(self, filename: str) -> str:
        if filename not in self.files:
            raise FileNotFoundError(f"{filename} not found in {self.folder_path}")
        path = os.path.join(self.folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # 3. return a specific file as a string by order in the folder
    def get_file_by_index(self, index: int) -> str:
        if index < 0 or index >= len(self.files):
            raise IndexError("Index out of range")
        filename = self.files[index]
        return self.get_file_by_name(filename)

    # 4. return a specific train input and output pair in the json by order
    def get_train_pair(self, filename: str, index: int) -> tuple[list, list]:
        data = self._load_file(filename)
        if index < 0 or index >= len(data["train"]):
            raise IndexError("Train index out of range")
        entry = data["train"][index]
        return entry["input"], entry["output"]

    # 5. return the test input
    def get_test_input(self, filename: str, index: int = 0) -> list:
        data = self._load_file(filename)
        if index < 0 or index >= len(data["test"]):
            raise IndexError("Test index out of range")
        return data["test"][index]["input"]

    # 6. return the test output
    def get_test_output(self, filename: str, index: int = 0) -> list:
        data = self._load_file(filename)
        if index < 0 or index >= len(data["test"]):
            raise IndexError("Test index out of range")
        return data["test"][index]["output"]
