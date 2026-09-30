"""Unit tests for Ollama utility functions."""
from unittest.mock import MagicMock, patch
import pytest

from src.ollama_utils import ensure_model_is_pulled, pull_model_with_indicator


class TestOllamaUtils:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        return MagicMock()

    def test_pull_model_with_indicator_streams_and_configures_tqdm(
        self, mock_client: MagicMock
    ) -> None:
        """Verify pulling streams updates and configures tqdm with leave=False."""
        update1 = MagicMock(digest="sha256:part1", completed=1024)
        update2 = MagicMock(digest="sha256:part2", completed=2048)
        update3 = MagicMock(digest=None, completed=None)
        mock_client.pull.return_value = [update1, update2, update3]

        with patch("src.ollama_utils.tqdm") as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_pbar

            pull_model_with_indicator("test-model", client=mock_client)

            mock_client.pull.assert_called_once_with(
                "test-model", stream=True
            )
            mock_tqdm.assert_called_once()
            _, kwargs = mock_tqdm.call_args
            assert kwargs.get("leave") is False
            assert kwargs.get("unit") == "B"
            assert kwargs.get("unit_scale") is True
            assert "test-model" in kwargs.get("bar_format", "")
            assert mock_pbar.n == 1024 + 2048

    def test_pull_model_with_indicator_uses_default_client_when_none(
        self,
    ) -> None:
        """Verify default client is used when client argument is None."""
        with patch("src.ollama_utils.ollama.Client") as mock_client_cls, \
             patch("src.ollama_utils.tqdm"):
            mock_instance = MagicMock()
            mock_instance.pull.return_value = []
            mock_client_cls.return_value = mock_instance

            pull_model_with_indicator("test-model", client=None)

            mock_client_cls.assert_called_once()
            mock_instance.pull.assert_called_once_with(
                "test-model", stream=True
            )

    def test_ensure_model_is_pulled_skips_when_exact_match(
        self, mock_client: MagicMock
    ) -> None:
        """Ensure pull is skipped when model name matches exactly."""
        mock_model = MagicMock(model="all-minilm:l6-v2")
        mock_client.list.return_value.models = [mock_model]

        with patch("src.ollama_utils.pull_model_with_indicator") as mock_pull:
            ensure_model_is_pulled("all-minilm:l6-v2", client=mock_client)
            mock_pull.assert_not_called()

    def test_ensure_model_is_pulled_skips_when_latest_tag_present(
        self, mock_client: MagicMock
    ) -> None:
        """Ensure pull is skipped when model has ':latest' suffix."""
        mock_model = MagicMock(model="qwen3:0.6b:latest")
        mock_client.list.return_value.models = [mock_model]

        with patch("src.ollama_utils.pull_model_with_indicator") as mock_pull:
            ensure_model_is_pulled("qwen3:0.6b", client=mock_client)
            mock_pull.assert_not_called()

    def test_ensure_model_is_pulled_skips_case_insensitively(
        self, mock_client: MagicMock
    ) -> None:
        """Ensure pull is skipped when case differs (e.g. Qwen3 vs qwen3)."""
        mock_model = MagicMock(model="qwen3:0.6b")
        mock_client.list.return_value.models = [mock_model]

        with patch("src.ollama_utils.pull_model_with_indicator") as mock_pull:
            ensure_model_is_pulled("Qwen3:0.6b", client=mock_client)
            mock_pull.assert_not_called()

    def test_ensure_model_is_pulled_triggers_pull_when_missing(
        self, mock_client: MagicMock
    ) -> None:
        """Ensure pull is called when model is not available."""
        mock_model = MagicMock(model="other-model:latest")
        mock_client.list.return_value.models = [mock_model]

        with patch("src.ollama_utils.pull_model_with_indicator") as mock_pull:
            ensure_model_is_pulled("target-model", client=mock_client)
            mock_pull.assert_called_once_with(
                "target-model", client=mock_client
            )
