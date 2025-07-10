from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer 
import numpy as np
import faiss

# 1. Extract text from PDF
reader = PdfReader("F:\\ml\\Attention.pdf")
text = ""
for page in reader.pages:
    text_in_page = page.extract_text()
    if text_in_page:
        text += text_in_page + "\n"

nltk.download('punkt')

# 2. Chunk text (paragraphs, fallback to sentences if too few)
chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 30]
if len(chunks) < 5:
    print("Few paragraph chunks found, falling back to sentence chunking.")
    chunks = [c.strip() for c in sent_tokenize(text) if len(c.strip()) > 10]

print(f"{len(set(chunks))} unique chunks out of {len(chunks)} total chunks.")

if len(chunks) == 0:
    print("No text found in PDF after chunking. Please check your PDF or extraction method.")
    exit()

# 3. Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
arr = np.array(embeddings).astype('float32')
dim = arr.shape[1]

# 4. Query embedding
query = "What is the main innovation used in the 'Attention is All You Need' Paper?"
embed = model.encode([query]).astype('float32')

# 5. Use IVF if enough chunks, else use Flat
nlist = 5
use_ivf = len(chunks) >= nlist

if use_ivf:
    print(f"Using IVF index with nlist={nlist}")
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(arr)
    index.add(arr)
    index.nprobe = min(5, nlist)
else:
    print("Too few chunks for IVF, using Flat index.")
    index = faiss.IndexFlatL2(dim)
    index.add(arr)

# 6. Search
k = min(5, len(chunks))
D, I = index.search(embed, k)

# 7. Gather top results for context
top_chunks = []
seen = set()
for idx, id in enumerate(I[0]):
    sentence = chunks[id]
    if sentence not in seen:
        top_chunks.append(sentence)
        seen.add(sentence)

context = "\n".join(top_chunks)

# 8. Build prompt for Phi-2
prompt = (
    "Instruct: Using the following information, answer the question.\n\n"
    f"{context}\n\n"
    "Question: What is the main innovation used in the 'Attention is All You Need' Paper?\n\n"
    "Output:\n"
)

# 9. Run Phi-2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nPhi-2's Answer:\n", result)
