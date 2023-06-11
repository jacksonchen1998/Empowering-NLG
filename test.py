from transformers import GPT2Tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(gpt_tokenizer(' the'))
