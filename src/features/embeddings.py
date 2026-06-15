import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Pinned revision SHA for InstaDeepAI/nucleotide-transformer-500m-1000g.
# This prevents supply-chain attacks (CWE-494 / Bandit B615) by ensuring the
# model weights are immutable and verified rather than fetched from a floating
# 'main' branch pointer.
_NUCLEOTIDE_TRANSFORMER_REVISION = "29bea3afa72fbd72ca75f80acbcb6b5be6d1a3ef"

class PretrainedEmbedder:
    """
    Utility class to load a pre-trained genomic language model from HuggingFace
    and extract sequence embeddings.
    """
    def __init__(self, model_name: str = "InstaDeepAI/nucleotide-transformer-500m-1000g",
                 revision: str = _NUCLEOTIDE_TRANSFORMER_REVISION,
                 device: str = None):
        self.model_name = model_name
        self.revision = revision
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading tokenizer and model: {model_name}@{revision} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)  # nosec B615
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(self.device)  # nosec B615
        self.model.eval()

    def get_embeddings(self, sequences: list, batch_size: int = 16) -> np.ndarray:
        """
        Extract embeddings for a list of DNA sequences.
        By default, we mean-pool the token embeddings to create a single vector per sequence.
        """
        all_embeddings = []
        
        # Determine mixed precision context based on device
        if "cuda" in self.device:
            context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting Embeddings (bs={batch_size})"):
                batch_seqs = sequences[i:i + batch_size]
                
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_seqs,
                    padding=True,
                    truncation=True,
                    max_length=1000,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass with mixed precision for speed
                with context:
                    outputs = self.model(**inputs)
                
                # Get the last hidden states: shape (batch_size, seq_len, hidden_size)
                hidden_states = outputs.last_hidden_state
                
                # Mask out padding tokens for mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                
                # Mean pool over the sequence length
                mean_pooled = sum_embeddings / sum_mask
                
                all_embeddings.append(mean_pooled.cpu().numpy())
                
        return np.vstack(all_embeddings)
