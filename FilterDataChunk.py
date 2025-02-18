from langflow.custom import Component
from langflow.io import DataInput, MessageTextInput, Output
from langflow.schema import Data
from typing import List

class CustomFilterDataComponent(Component):
    display_name = "Filter Custom Data"
    description = "Filters a Data object and removes prefixes from text, outputs a list of Data objects."
    icon = "filter"
    beta = True
    name = "FilterData"

    inputs = [
        DataInput(
            name="data",
            display_name="Data",
            info="Data object to filter.",
        )
    ]

    outputs = [
        Output(display_name="Filtered Data", name="filtered_data", method="filter_data"),
    ]

    def filter_data(self) -> List[Data]:
        filtered_list_data = []
        raw_data = self.data.data if isinstance(self.data, Data) else self.data


        prefixes_to_remove = ["Type: program\nContent:\n", "Type: general_info\nContent:\n"]

        if isinstance(raw_data, list) and len(raw_data) > 0:
            for item in raw_data:
                item_data = item.data
                if isinstance(item_data, dict) and "data" in item_data and isinstance(item_data["data"], dict):
                    pk = item_data["data"].get("pk")
                    text = item_data["data"].get("text", "")
                    for prefix in prefixes_to_remove:
                        if text.startswith(prefix):
                            text = text[len(prefix):] # Remove prefix
                    for sep in ["Từ khoá:", "Metadata:"]:
                        if sep in text:
                            text = text.split(sep)[0].strip()
                    result = {"pk": pk, "text": text}
                    filtered_list_data.append(Data(data=result))
                elif isinstance(item_data, dict) and "pk" in item_data and "text" in item_data:
                    pk = item_data.get("pk")
                    text = item_data.get("text", "")
                    for prefix in prefixes_to_remove:
                        if text.startswith(prefix):
                            text = text[len(prefix):] # Remove prefix
                    for sep in ["Từ khoá:", "Metadata:"]:
                        if sep in text:
                            text = text.split(sep)[0].strip()
                    result = {"pk": pk, "text": text}

                    filtered_list_data.append(Data(data=result))
                elif isinstance(item_data, str):
                    text = item_data
                    for prefix in prefixes_to_remove:
                        if text.startswith(prefix):
                            text = text[len(prefix):] # Remove prefix
                    for sep in ["Từ khoá:", "Metadata:"]:
                        if sep in text:
                            text = text.split(sep)[0].strip()
                    result = {"text": text}
                    filtered_list_data.append(Data(data=result))

        elif isinstance(raw_data, dict):
            if "pk" in raw_data and "text" in raw_data:
                pk = raw_data.get("pk")
                text = raw_data.get("text", "")
                for prefix in prefixes_to_remove:
                    if text.startswith(prefix):
                        text = text[len(prefix):] # Remove prefix
                for sep in ["Từ khoá:", "Metadata:"]:
                    if sep in text:
                        text = text.split(sep)[0].strip()
                result = {"pk": pk, "text": text}

                filtered_list_data.append(Data(data=result))
            else:

                filtered_list_data.append(Data(data=raw_data))
        elif isinstance(raw_data, str):
            text = raw_data
            for prefix in prefixes_to_remove:
                if text.startswith(prefix):
                    text = text[len(prefix):] # Remove prefix
            for sep in ["Từ khoá:", "Metadata:"]:
                if sep in text:
                    text = text.split(sep)[0].strip()
            result = {"text": text}

            filtered_list_data.append(Data(data=result))

        self.status = filtered_list_data
        return filtered_list_data