import numpy as np
import torch
import warnings

SEED = 42

torch.set_default_dtype(torch.float32)
warnings.filterwarnings('ignore')
# Load ESM-2 only when embeddings are requested (CLI help stays lightweight).
device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else
                      'cuda:0' if torch.cuda.is_available() else 'cpu')
esm_model = None
batch_converter = None


def _initialize_model():
    global esm_model, batch_converter
    if esm_model is None:
        import esm.pretrained
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_model = esm_model.to(device)
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()


# get ESM2 embedding
def get_ESM_embeddings(sequence):
    _initialize_model()
    esm_input = [('_', sequence)]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(esm_input)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_embeddings = results["representations"][33]
    token_embeddings = token_embeddings[:, 1:-1].squeeze().cpu().numpy()
    
    return token_embeddings
# output: array, size = [num_residues, 1280]

# load ESM2 embedding
def load_embedding(embedding_path):
    embedding = np.load(embedding_path)
    
    return embedding