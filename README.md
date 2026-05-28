# llm-psychology

A small library for training LLMs on neuroscience / psychology tasks, inspecting their representations, and modelling memory encoding / retrieval / consolidation.

Install it with:
```bash
pip install .
```

> [!NOTE]
> For code to reproduce results in Spens & Burgess (2026), see [`ellie-as/hippocampal-neocortical-RAG`](https://github.com/ellie-as/hippocampal-neocortical-RAG) instead.

## Tutorial notebooks

See the following notebooks for tutorials:
- `notebooks/01_statistical_learning_example.ipynb`: This trains GPT-2 from scratch on the stimuli from Durrant et al. (2011), to demonstrate how to train a small LLM on youe own data, and then inspects the learned attention.
- `notebooks/02_visualisation_features.ipynb`: This demonstrates how to inspect the token representations at different layers of the model, using the example of models trained on spatial / family tree inference in Spens & Burgess (2026).
- `notebooks/03_rag_local_memory.ipynb`: This is simple demo of retrieval-augmented generation, showing how parametric (in-weights) and non-parametric (in-context) forms of memory can be combined. 
- `notebooks/04_xrag_compression_rocstories.ipynb`: This shows how to compress 'memories' with xRAG before encoding them, then 'recall' memories from the compressed representations.

## Supported features

### Train from text files

Train end-to-end:
```python
from llm_psychology.train.trainer import train_causal_lm

model, tok, _ = train_causal_lm(
    model_name="gpt2",                   # or a local checkpoint
    train_path="path/to/train.txt",
    eval_path="path/to/val.txt",
    output_dir="./outputs/run",
    max_steps=50,
    # from_scratch=True, n_layer=4, n_head=4, n_embd=256,  # optional
)
```

Train with LoRA, a parameter-efficient fine-tuning approach (this allows larger LLMs to be trained, e.g. Mistral 7B on a single A100):
```python
model, tok, metrics = train_causal_lm(
    model_name="gpt2",
    train_path="train.txt",
    eval_path="val.txt",
    output_dir="./outputs/lora_run",
    peft_method="lora",
    lora_r=8,
    max_steps=100,
)
```

### Visualise representations and attention

Do PCA on token representations to shed light on the structure learned by the model:
```python
from llm_psychology.reps.extractor import extract_hidden_states, extract_token_representations
from llm_psychology.viz.pca import pca_by_layer

hs, _ = extract_hidden_states(model, tok, ["some text"])
fig = pca_by_layer(hs.numpy(), layer_indices=[0, hs.shape[1]-1])
fig.savefig("pca.png", dpi=150)

token_df = extract_token_representations(model, tok, ["1 2 3 4"], layers=[0, -1])
print(token_df[["token", "layer"]].head())
```

See what the model attends to in the input:
```python
from llm_psychology.reps.extractor import extract_attention_weights
from llm_psychology.viz.attention import plot_attention_heatmap

attn, tokens, _ = extract_attention_weights(model, tok, ["1 2 3 4"])
fig = plot_attention_heatmap(attn, tokens[0], layer=-1)
```

### xRAG Compression
```python
from llm_psychology.xrag import (
    XRAG,
    XRAGConfig,
)

xrag = XRAG(XRAGConfig())  # Hannibal046/xrag-7b + Salesforce/SFR-Embedding-Mistral
xrag.load()
datastore = xrag.prepare_datastore(stories["text"].tolist())
answer = xrag.answer("Melody went to the aquarium... What happened next?", datastore)
print(answer.text)
```
