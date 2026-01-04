import pytest
import os
from typing import Any
from unittest.mock import patch


def test_config_validation_local_storage(mocker: Any) -> None:
    # If using local storage, R2 keys are not required
    with patch.dict(os.environ, {
        "USE_LOCAL_STORAGE": "true",
        "R2_ENDPOINT_URL": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": ""
    }):
        # Reload module to pick up env vars
        import config.settings
        import importlib
        importlib.reload(config.settings)
        from config.settings import Settings
        
        # Should not raise
        Settings.validate()

def test_config_validation_r2_missing_keys(mocker: Any) -> None:
    # If using R2, keys are required
    with patch.dict(os.environ, {
        "USE_LOCAL_STORAGE": "false",
        "R2_ENDPOINT_URL": "", 
        # Missing keys
    }):
        import config.settings
        import importlib
        importlib.reload(config.settings)
        from config.settings import Settings
        
        with pytest.raises(ValueError, match="R2 credentials are required"):
            Settings.validate()

def test_validation_api_key_check(capsys: Any) -> None:
    # Check if warning is printed when Google API key is missing
    with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
        import config.settings
        import importlib
        importlib.reload(config.settings)
        from config.settings import Settings
        
        Settings.validate()
        captured = capsys.readouterr()
        assert "WARNING: GOOGLE_API_KEY is missing" in captured.out
