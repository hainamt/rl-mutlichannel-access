class QTable(dict):
    def __init__(self, channel_objects):
        super().__init__()
        self.channel_objects = channel_objects
        self.channel_map = {ch.channel_index: ch for ch in channel_objects}

    def _convert_key(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            t, c = key
            if isinstance(c, int) and c in self.channel_map:
                return t, self.channel_map[c]
        return key

    def __getitem__(self, key):
        converted_key = self._convert_key(key)
        return super().__getitem__(converted_key)

    def __setitem__(self, key, value):
        converted_key = self._convert_key(key)
        return super().__setitem__(converted_key, value)

    def __contains__(self, key):
        converted_key = self._convert_key(key)
        return super().__contains__(converted_key)
