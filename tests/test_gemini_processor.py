import pytest
from unittest.mock import MagicMock
from src.processors.gemini_processor import GeminiProcessor

def test_gemini_init_missing_key(mocker):
    # Mock settings.GOOGLE_API_KEY as None
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', None)
    
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
        GeminiProcessor()

def test_gemini_refine_transcript(mocker):
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mock_genai = mocker.patch('src.processors.gemini_processor.genai')
    
    # Mock model
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    # Mock generate_content
    mock_response = MagicMock()
    mock_response.text = "Refined Text"
    mock_model.generate_content.return_value = mock_response
    
    processor = GeminiProcessor()
    result = processor.refine_transcript("Raw Text")
    
    assert result == "Refined Text"
    mock_model.generate_content.assert_called_once()
    # Could verify prompt contains "Raw Text"

def test_gemini_refine_retry_logic(mocker):
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mock_genai = mocker.patch('src.processors.gemini_processor.genai')
    mock_model = mock_genai.GenerativeModel.return_value
    
    # First call failed, second succeeded
    mock_response = MagicMock()
    mock_response.text = "Refined Text"
    
    mock_model.generate_content.side_effect = [Exception("API Error"), mock_response]
    
    processor = GeminiProcessor()
    # Patch time.sleep to avoid waiting
    mocker.patch('time.sleep') 
    
    result = processor.refine_transcript("Raw")
    assert result == "Refined Text"
    assert mock_model.generate_content.call_count == 2

def test_gemini_get_embedding(mocker):
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mock_genai = mocker.patch('src.processors.gemini_processor.genai')
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2]}

    processor = GeminiProcessor()

    # Ensure configure called
    mock_genai.configure.assert_called()

    result = processor.get_embedding("text")

    assert result == [0.1, 0.2]
    mock_genai.embed_content.assert_called_once()

def test_gemini_embedding_uses_correct_model(mocker):
    """임베딩 생성 시 gemini-embedding-001 모델을 사용하는지 검증"""
    mocker.patch('src.processors.gemini_processor.settings.GOOGLE_API_KEY', "fake_key")
    mock_genai = mocker.patch('src.processors.gemini_processor.genai')
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2]}

    processor = GeminiProcessor()
    processor.get_embedding("테스트 텍스트")

    # 올바른 모델명으로 호출되었는지 검증
    call_args = mock_genai.embed_content.call_args
    assert call_args.kwargs['model'] == "models/gemini-embedding-001", \
        f"Expected 'models/gemini-embedding-001' but got '{call_args.kwargs['model']}'"
