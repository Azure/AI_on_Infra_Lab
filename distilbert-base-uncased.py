import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import time

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

if __name__ == "__main__":
    model_name = 'distilbert-base-uncased'
    dataset_name = 'imdb'
    benchmark(model_name, dataset_name)

