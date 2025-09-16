# tests/test_utils.py
# run with pytest tests/.
# 
# import pytest
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

from utils.logger import get_logger, setup_logging
from utils.utils import is_dict, is_list_of_dicts
from utils.config import TITLE, DESCRIPTION  # Assuming these are defined
from utils.get_config import get_config_value  # If separate module

def test_setup_logging(capsys):
    setup_logging()
    captured = capsys.readouterr()
    assert "Logging configured" in captured.out or captured.err  # Assuming it prints config message

def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

@patch('logging.getLogger')
def test_get_logger_custom(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    logger = get_logger("custom_test")
    mock_get_logger.assert_called_once_with("custom_test")
    assert logger == mock_logger

def test_is_dict():
    assert is_dict({"key": "value"}) is True
    assert is_dict({"key": [1, 2]}) is True
    assert is_dict([]) is False
    assert is_dict("string") is False
    assert is_dict(123) is False
    assert is_dict(None) is False

def test_is_list_of_dicts():
    assert is_list_of_dicts([{"a": 1}, {"b": 2}]) is True
    assert is_list_of_dicts([]) is False  # Empty list not considered list of dicts
    assert is_list_of_dicts([{"a": 1}, "string"]) is False
    assert is_list_of_dicts("not_list") is False
    assert is_list_of_dicts([1, 2]) is False
    assert is_list_of_dicts(None) is False

def test_config_constants():
    # Test if config values are as expected (update based on actual config.py)
    assert TITLE == "parserPDF"  # Or whatever the actual value is
    assert DESCRIPTION.startswith("PDF parser")  # Partial match for description

@patch('utils.get_config.configparser.ConfigParser')
def test_get_config_value(mock_configparser):
    mock_config = MagicMock()
    mock_config.get.return_value = "test_value"
    mock_configparser.return_value = mock_config
    
    value = get_config_value("SECTION", "KEY")
    mock_config.get.assert_called_once_with("SECTION", "KEY")
    assert value == "test_value"

@patch('utils.get_config.configparser.ConfigParser')
def test_get_config_value_default(mock_configparser):
    mock_config = MagicMock()
    mock_config.get.side_effect = KeyError("No such key")
    mock_configparser.return_value = mock_config
    
    value = get_config_value("SECTION", "NONEXISTENT", default="fallback")
    assert value == "fallback"
    mock_config.get.assert_called_once_with("SECTION", "NONEXISTENT")

def test_logger_levels(caplog):
    # Test logging at different levels
    logger = get_logger("level_test")
    
    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug message")
        assert "Debug message" in caplog.text
    
    with caplog.at_level(logging.INFO):
        logger.info("Info message")
        assert "Info message" in caplog.text
    
    with caplog.at_level(logging.ERROR):
        logger.error("Error message")
        assert "Error message" in caplog.text

def test_setup_logging_file(tmp_path):
    log_file = tmp_path / "test.log"
    with patch.dict('os.environ', {'LOG_FILE': str(log_file)}):
        setup_logging()
        assert log_file.exists()
        log_file.unlink()  # Cleanup