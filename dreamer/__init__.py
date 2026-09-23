import os

# litellm calls load_dotenv() on import unless LITELLM_MODE != "DEV". That
# pushes the base .env into os.environ before config.load_config runs, where
# it is indistinguishable from the shell environment and so beats the
# .env.<environment> overlay. Config loading belongs to config.py alone.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
