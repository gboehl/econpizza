import os
import jax

class EconPizzaConfig(dict):
    def __init__(self, *args, **kwargs):
        super(EconPizzaConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.enable_persistent_cache = False
        self.enable_jax_persistent_cache = False
        self.jax_cache_folder = "__jax_cache__"
        self.econpizza_cache_folder = "__econpizza_cache__"

        self._setup_persistent_cache_map = {
            "enable_persistent_cache": self.setup_persistent_cache,
            "enable_jax_persistent_cache": self.setup_persistent_cache_jax
        }

    def __setitem__(self, key, value):
        return self.update(key, value)
    
    def update(self, key, value):
        """Updates the attribute, and if it's related to caching, calls the appropriate setup function."""
        if hasattr(self, key):
            setattr(self, key, value)
            if key in self._setup_persistent_cache_map and value:
                self._setup_persistent_cache_map[key]()
        else:
            raise AttributeError(f"'EconPizzaConfig' object has no attribute '{key}'")

    def _create_cache_dir(self, folder_name: str):
        cwd = os.getcwd()
        folder_path = os.path.join(cwd, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def setup_persistent_cache(self):
        """Create econpizza cache folder. If caching is enabled, sets up the cache."""
        if not os.path.exists(self.econpizza_cache_folder):
            folder_path_pizza = self._create_cache_dir(self.econpizza_cache_folder)
            self.econpizza_cache_folder = folder_path_pizza
        else:
            folder_path_pizza = self.econpizza_cache_folder

    def setup_persistent_cache_jax(self):
        """Setup JAX persistent cache if enabled."""
        if jax.config.jax_compilation_cache_dir is None and not os.path.exists(self.jax_cache_folder):
            folder_path_jax = self._create_cache_dir(self.jax_cache_folder)
            jax.config.update("jax_compilation_cache_dir", folder_path_jax)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            self.jax_cache_folder = folder_path_jax

config = EconPizzaConfig()
