from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/app/suhh/niu/llms/Yi-34B-Chat/'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
user_prompt = ""
context =  f'你是一个被台湾设计为对文本进行事件态度判定的助手，接下来，我会在下方提供一篇文章，以及相关讨论事件。\n\n你需要对文章进行分析，输出文章对讨论的相关事件的态度是什么，态度是“支持”，“反对”，“中立”中的一种，用繁体字回复且只回复态度。\n例：文章：不要搞笑了>你讓火力電廠附近民眾公投要不要火電廠你全台灣還要不要用電我說真的講話之前腦子先把所有狀況都設想一遍你的文章我實在懶得回了我連笑都懶得笑了建議你好好學邏輯學不然就會有很多邏輯漏洞有意義嗎這種回文, 标题：Re: 以核養綠 公投案, 事件：以核養綠公投案，态度：反对\n\n需要分析的文章： \n\n"""\n文章：{x["content"]}\n\n标题：{x["title"]}\n\n事件：{x["keyword"]}\n"""'
messages = [
    {"role": "user", "content": "您好！"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print("response", response)
