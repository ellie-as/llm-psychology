from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from .retrieve import Retriever
from ..models.generate import simple_generate


def rag_answer(retriever: Retriever, lm_path_or_name: str, queries: List[str],
               k: int = 3, prefix: str = "Context:\n{context}\n\nQuestion: {q}\nAnswer:",
               do_sample: bool = False) -> List[str]:
    model = AutoModelForCausalLM.from_pretrained(lm_path_or_name, use_safetensors=True, low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(lm_path_or_name)

    outputs: List[str] = []
    for q in queries:
        rets = retriever.retrieve([q], top_k=k)[0]
        ctx = "\n".join([t for _, t in rets])
        prompt = prefix.format(context=ctx, q=q)
        outputs.extend(simple_generate(model, tok, [prompt], max_new_tokens=64, do_sample=do_sample))
    return outputs


