import torch
import unittest
import numpy as np

class TestTorchScriptModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the TorchScript model
        cls.model = torch.jit.load('notebooks/simple_model_scripted.pt')
        cls.model.eval()  # Set the model to evaluation mode

    def test_model_output_shape(self):
        """Test if the model outputs the correct shape."""
        input_tensor = torch.randn(1, 5)  # Adjust shape based on model input requirements
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (1, 5), "Output shape mismatch")

    def test_model_output_values(self):
        """Test if the model output values are within an expected range."""
        input_tensor = torch.randn(1, 5)
        output_tensor = self.model(input_tensor)
        # Example: Check if all output values are within the range -1 to 1
        self.assertTrue(torch.all(output_tensor >= -1) and torch.all(output_tensor <= 1),
                        "Output values out of expected range")

    def test_model_with_different_inputs(self):
        """Test the model with various types of inputs to ensure robustness."""
        inputs = [
            torch.zeros(1, 5),
            torch.ones(1, 5),
            torch.randn(1, 5),
            torch.full((1, 5), 0.5)
        ]
        for input_tensor in inputs:
            output_tensor = self.model(input_tensor)
            self.assertEqual(output_tensor.shape, (1, 5), "Output shape mismatch with different inputs")

    def test_model_gradients(self):
        """Test if the model's gradients are computed correctly."""
        input_tensor = torch.randn(1, 5, requires_grad=True)
        output_tensor = self.model(input_tensor)
        output_tensor.sum().backward()
        self.assertIsNotNone(input_tensor.grad, "Gradients were not computed")

    def test_scripted_model_serialization(self):
        """Test if the scripted model can be reloaded and produce consistent outputs."""
        input_tensor = torch.randn(1, 5)
        output_original = self.model(input_tensor)

        # Save and reload the scripted model
        torch.jit.save(self.model, 'test_scripted_model.pt')
        reloaded_model = torch.jit.load('test_scripted_model.pt')
        reloaded_model.eval()

        output_reloaded = reloaded_model(input_tensor)
        self.assertTrue(torch.allclose(output_original, output_reloaded),
                        "Outputs differ after reloading the scripted model")

if __name__ == '__main__':
    unittest.main()
