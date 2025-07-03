import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pretrained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-T5-738M")
tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")

# Set the model to evaluation mode
model.eval()

# Prepare the model for quantization
model = torch.quantization.quantize_dynamic(
    model,  # the model to quantize
    {torch.nn.Linear},  # layers to quantize
    dtype=torch.qint8  # quantization type
)

torch.save(model, "./quantized_lamini_t5.pth")
print("model saved successfully")
