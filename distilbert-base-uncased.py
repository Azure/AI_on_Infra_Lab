import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import time
import argparse

def benchmark(model_name, dataset_name, batch_size=8):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to('cuda')

    # Load dataset
    dataset = load_dataset(dataset_name, split='test[:10%]')

    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Prepare dataloader
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)

    # Benchmarking
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total time for processing the dataset: {total_time:.2f} seconds')
    print(f'Average time per batch: {total_time / len(dataloader):.2f} seconds')

def interpret_prediction(prediction):
    sentiment=["Negative Sentiment", "Positive Sentiment"]
    return sentiment[prediction[0]]

def run_inference(model_name, sample_text):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to('cuda')

    # Tokenize the sample text
    inputs = tokenizer(sample_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark and run inference using a pretrained model")
    parser.add_argument("--sample_text", type=str, required=True, help="Sample text for inference")

    args = parser.parse_args()

    model_name = 'distilbert-base-uncased'
    dataset_name = 'imdb'

    # Run benchmark
    benchmark(model_name, dataset_name)

    # Run sample inference
    prediction = run_inference(model_name, args.sample_text)
    sentiment = interpret_prediction(prediction)
    print(f'Prediction for sample text: {prediction} ({sentiment})')
