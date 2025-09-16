# tests/test_llm.py
# run with pytest tests/.
# 
# import pytest
from unittest.mock import patch, MagicMock
import huggingface_hub
from huggingface_hub import get_token

from llm.llm_login import login_huggingface
from llm.provider_validator import is_valid_provider, suggest_providers
from llm.hf_client import HFChatClient  # Assuming this exists
from llm.openai_client import OpenAIClient  # Assuming this exists

def test_login_huggingface_success():
    with patch('huggingface_hub.login') as mock_login:
        api_token = "hf_test_token"
        login_huggingface(api_token)
        mock_login.assert_called_once_with(token=api_token, add_to_git_credential=False)

def test_login_huggingface_no_token():
    with patch('huggingface_hub.login') as mock_login:
        with pytest.raises(ValueError, match="API token required"):
            login_huggingface(None)

@patch('huggingface_hub.login')
def test_login_huggingface_error(mock_login):
    mock_login.side_effect = Exception("Login failed")
    with pytest.raises(Exception, match="Login failed"):
        login_huggingface("invalid_token")

def test_is_valid_provider():
    assert is_valid_provider("huggingface") is True
    assert is_valid_provider("openai") is True
    assert is_valid_provider("invalid_provider") is False
    assert is_valid_provider("") is False
    assert is_valid_provider(None) is False

def test_suggest_providers():
    suggestions = suggest_providers("hugngface")  # Typo example
    assert isinstance(suggestions, list)
    assert "huggingface" in suggestions
    
    no_suggestions = suggest_providers("completely_unknown")
    assert isinstance(no_suggestions, list)
    assert len(no_suggestions) == 0

@patch('llm.hf_client.HFChatClient.__init__')
def test_hf_client_init(mock_init):
    mock_init.return_value = None
    client = HFChatClient(model_id="test-model", api_token="test_token")
    mock_init.assert_called_once_with(
        model_id="test-model",
        api_token="test_token",
        # Add other expected params based on actual __init__
    )

@patch('llm.hf_client.login_huggingface')
@patch('llm.hf_client.get_token')
def test_hf_client_token(mock_get_token, mock_login):
    mock_get_token.return_value = "cached_token"
    mock_login.return_value = None
    
    client = HFChatClient(model_id="test-model")
    assert client.api_token == "cached_token"

@patch('openai.OpenAI')
def test_openai_client_init(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    client = OpenAIClient(api_key="sk_test_key", base_url="https://api.openai.com/v1")
    mock_openai.assert_called_once_with(
        api_key="sk_test_key",
        base_url="https://api.openai.com/v1"
    )
    assert client.client == mock_client

@patch('openai.OpenAI')
def test_openai_client_chat(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(content="Hello!")]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    client = OpenAIClient(api_key="sk_test_key")
    response = client.chat("Hello", model="gpt-3.5-turbo")
    
    assert response == "Hello!"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )

def test_provider_validator_edge_cases():
    # Test with non-string inputs
    assert is_valid_provider(123) is False
    assert suggest_providers(123) == []
    
    # Test case insensitivity
    assert is_valid_provider("HUGGINGFACE") is True
    assert is_valid_provider("OpEnAi") is True

@patch('huggingface_hub.get_token')
def test_get_token_from_env(mock_get_token):
    mock_get_token.return_value = None
    with patch.dict('os.environ', {'HUGGINGFACE_HUB_TOKEN': 'env_token'}):
        token = get_token()
        assert token == 'env_token'

@patch('huggingface_hub.get_token')
def test_get_token_from_cache(mock_get_token):
    mock_get_token.return_value = 'cached_token'
    token = get_token()
    assert token == 'cached_token'