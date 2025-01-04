# Qwen-
## Qwen流式响应以及联网检索
1. 请本地部署您的qwen大模型, 本人选用了7B和14B。不论您选用了什么样的模型，只要是被transformers封装过的大模型都可以使用；
2. 打开`stream_chat.ipynb`文件，该文件提供了使用样例。
3. 您还可以在`LLM.py`文件中添加您想要的任意工具。

## 用法
1. 您可以初始化LLM.py中的myLLM类, 传入模型路径、分词器路径、并定义是否联网(website_search参数);
2. 进一步使用`set_system`方法定义大模型的角色;
4. 调用`stream_chat`方法, 传入query就可以使用啦。
