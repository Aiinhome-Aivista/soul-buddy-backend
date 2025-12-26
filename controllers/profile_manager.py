import os
import json

class ProfileManager:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_file_path(self, user_id):
        return os.path.join(self.storage_path, f"{user_id}.json")

    def load_profile(self, user_id):
        path = self._get_file_path(user_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def save_profile(self, user_id, profile_data):
        path = self._get_file_path(user_id)
        with open(path, "w") as f:
            json.dump(profile_data, f, indent=2)
