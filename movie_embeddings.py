# import necessary library for using the model
from transformers import BertTokenizer, BertModel
import get_data
import torch
import numpy as np
from tqdm import tqdm

# establishing the tokenizer and the model

EMBEDDINGS_FILE = 'movie_embeddings.npy'
BATCH_SIZE =256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def generate_movie_embeddings():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    df = get_data.get_data()
    all_embeddings = []

    # Tokenize the text soup
    # The `return_tensors='pt'` argument returns PyTorch tensors
    # The `padding=True` pads all sequences to the same length
    # The `truncation=True` truncates sequences longer than the model's max length
    print("Generating embeddings in batches...")
    
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        # 1. Get a small batch of the dataframe
        df_batch = df.iloc[i:i + BATCH_SIZE]
        texts = df_batch['text_soup'].tolist()

        # 2. Tokenize just this batch (fast and low RAM)
        tokenized_text = tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # 3. Move this small batch to the GPU
        tokenized_text = tokenized_text.to(device)

        # 4. Generate embeddings for this batch
        with torch.no_grad():
            outputs = model(**tokenized_text)
        
        # 5. Get the embeddings, move to CPU, and add to our list
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(embeddings)

    movie_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    print(f"Shape of movie embeddings: {movie_embeddings.shape}")

    print(f"Saving the movie embeddings to {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, movie_embeddings)

    return movie_embeddings


 

generate_movie_embeddings()