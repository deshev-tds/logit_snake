class ModelConfiguration:
    def __init__(self, model, config_name, value, enable=True):
        self.model = model
        self.generation_config = model.get_generation_config('generation_config')
        self.config_name = config_name

        self.old_value = getattr(self.generation_config, self.config_name, None)
        self.new_value = value

        self.enable = enable

    def __enter__(self):
        if self.enable:
            self.model.set_generation_config(self.config_name, self.new_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            # recover the original setting
            self.model.set_generation_config(self.config_name, self.old_value)
