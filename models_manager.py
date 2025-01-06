sentece_transformer = None
def get_sentence_transformer_model():
    global sentece_transformer
    if sentece_transformer is not None:
        return sentece_transformer
    from sentence_transformers import SentenceTransformer
    sentece_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentece_transformer

caption_model = None 
caption_processor = None
caption_tokenizer = None

def get_caption_model():
    global caption_model, caption_processor, caption_tokenizer
    if caption_model is not None:
        return caption_model, caption_processor, caption_tokenizer
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return caption_model, caption_processor, caption_tokenizer

def get_embedding_model():
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor