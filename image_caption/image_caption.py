from models_manager import get_caption_model
def get_caption(image):
    model, processor, tokenizer = get_caption_model()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption