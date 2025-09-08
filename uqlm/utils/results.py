from typing import Dict, Any
import pandas as pd


class UQResult:
    def __init__(self, result: Dict[str, Any]) -> None:
        """
        Class that characterizes result of an UncertaintyQuantifier.

        Parameters
        ----------
        result: dict
            A dictionary that is defined during `evaluate` or `tune_params` method
        """
        self.data = result.get("data")
        self.metadata = result.get("metadata")
        self.result_dict = result

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns result in dictionary form
        """
        return self.result_dict

    def to_df(self) -> pd.DataFrame:
        """
        Returns result in pd.DataFrame
        """
        rename_dict = {col: col[:-1] for col in self.result_dict["data"].keys() if col.endswith("s") and col not in ["sampled_responses", "raw_sampled_responses"]}

        return pd.DataFrame(self.result_dict["data"]).rename(columns=rename_dict)
