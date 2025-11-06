"""model_handler.py: Transformer based model handler for time series forecasting."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerModelHandler:
    """
    Transformer based Model handler for time series forecasting.
    """

    def __init__(self, model_path: str = "hf_model"):
        """Initialize the model handler."""
        logger.info("Initializing Transformer Model Handler")
        self.model_path = model_path
        self.model = None
        self.config = None
        self.device = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Initialize model.

        Returns:
            bool: True if initialization successful
        """
        try:
            logger.info("Starting model initialization")

            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            logger.info(f"Loading model from: {self.model_path}")

            # Load model configuration
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

            # Load the pretrained model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, config=self.config, trust_remote_code=True
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info("Transformer model loaded successfully")
            logger.info(f"Model config: {self.config}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.initialized = False
            return False

    def preprocess(
        self,
        time_series_data: list,
        forecast_window: int = 24,
        input_len: int | None = None,
        frequency: str = "H",
        target_column: str = "value",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Transform raw input into model input data.

        Args:
            time_series_data: Input time series data
            forecast_window: Number of future points to predict
            input_len: Optional input length limit
            frequency: Frequency of the time series
            target_column: Column name for structured data

        Returns:
            Tuple of (processed_tensor, metadata)
        """
        try:
            logger.info(f"Input data length: {len(time_series_data) if isinstance(time_series_data, list) else 'N/A'}")
            logger.info(f"Forecast window: {forecast_window}")

            # Convert input to numpy array
            if isinstance(time_series_data, list):
                if len(time_series_data) > 0 and isinstance(time_series_data[0], dict):
                    # Handle structured data (with timestamps)
                    df = pd.DataFrame(time_series_data)
                    if target_column in df.columns:
                        values = df[target_column].values
                    else:
                        # Use the first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            values = df[numeric_cols[0]].values
                        else:
                            values = np.array([float(x) for x in time_series_data])
                else:
                    # Handle simple numeric list
                    values = np.array([float(x) for x in time_series_data])
            else:
                values = np.array(time_series_data)

            # Handle input length constraint
            if input_len is not None:
                if input_len > len(values):
                    # Pad with zeros if input is shorter than required
                    values = np.pad(values, (input_len - len(values), 0), mode="constant", constant_values=0)
                elif input_len < len(values):
                    # Take the last input_len values
                    values = values[-input_len:]

            # Normalize the data (simple z-score normalization)
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > 0:
                normalized_values = (values - mean_val) / std_val
            else:
                normalized_values = values - mean_val

            # Convert to tensor
            tensor = torch.tensor(normalized_values, dtype=torch.float32).unsqueeze(0)
            tensor = tensor.to(self.device)

            # Store metadata for post-processing
            metadata = {
                "mean": mean_val,
                "std": std_val,
                "forecast_window": forecast_window,
                "frequency": frequency,
                "original_length": len(values),
            }

            logger.info(f"Preprocessed tensor shape: {tensor.shape}")

            return tensor, metadata

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise e

    def inference(self, input_tensor: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        """
        Run inference on the model.

        Args:
            input_tensor: Preprocessed input tensor
            metadata: Preprocessing metadata

        Returns:
            Model output tensor
        """
        try:
            if not self.initialized:
                raise RuntimeError("Model not initialized")

            with torch.no_grad():
                forecast_window = metadata.get("forecast_window", 24)

                # Use generate method
                output = self.model.generate(input_tensor, max_new_tokens=forecast_window)

                logger.info(f"Model output shape: {output.shape}")
                return output

        except ValueError as e:
            logger.error(f"Inference failed due to ValueError: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise e

    def postprocess(self, output_tensor: torch.Tensor, metadata: dict[str, Any]) -> list[float]:
        """
        Transform model output to final prediction format.

        Args:
            output_tensor: Raw model output
            metadata: Preprocessing metadata

        Returns:
            Final predictions as list
        """
        try:
            # Extract predictions from tensor
            if output_tensor.dim() > 1:
                predictions = output_tensor[0].cpu().numpy()
            else:
                predictions = output_tensor.cpu().numpy()

            # Denormalize the predictions
            mean_val = metadata.get("mean", 0)
            std_val = metadata.get("std", 1)

            if std_val > 0:
                denormalized_predictions = predictions * std_val + mean_val
            else:
                denormalized_predictions = predictions + mean_val

            # Convert to list and ensure it's the right length
            forecast_window = metadata.get("forecast_window", 24)
            result = denormalized_predictions[:forecast_window].tolist()

            logger.info(f"Final predictions length: {len(result)}")

            return result

        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            raise e

    def predict(
        self,
        time_series_data: list,
        forecast_window: int = 24,
        input_len: int | None = None,
        frequency: str = "H",
        target_column: str = "value",
    ) -> dict[str, Any]:
        """
        Main prediction method.

        Args:
            time_series_data: Input time series data
            forecast_window: Number of future points to predict
            input_len: Optional input length limit
            frequency: Frequency of the time series
            target_column: Column name for structured data

        Returns:
            Dictionary containing predictions and metadata
        """
        try:
            # Ensure model is initialized
            if not self.initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize model")

            # Preprocess input
            input_tensor, metadata = self.preprocess(
                time_series_data, forecast_window, input_len, frequency, target_column
            )

            # Run inference
            output_tensor = self.inference(input_tensor, metadata)

            # Postprocess output
            predictions = self.postprocess(output_tensor, metadata)

            # Format result
            result = {
                "predictions": predictions,
                "forecast_window": metadata.get("forecast_window", 24),
                "frequency": metadata.get("frequency", "H"),
                "status": "success",
            }

            return result

        except ValueError as e:
            logger.error(f"Prediction failed due to ValueError: {str(e)}")
            return {"error": str(e), "code": 400, "status": "error"}
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e), "status": "error"}


# Global model handler instance
_model_handler = None


def get_model_handler() -> TransformerModelHandler:
    """Get or create global model handler instance."""
    global _model_handler
    if _model_handler is None:
        _model_handler = TransformerModelHandler()
    return _model_handler
