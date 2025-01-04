from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from lxml import etree
import numpy as np
import requests

class myLLM:
    def __init__(self, model_dir, tokenizer_dir, website_search=False):
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map='cuda:0', 
            trust_remote_code=True
        ).eval()

        self.streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.history = []
        self.website_search = website_search
    
    def set_system(self, text):
        self.history.append(
            {"role": "system", "content": text}
        )
    
    def _history_to_string(self, history):
        if len(history) == 0:
            return ''
        
        else:
            out = ''
            for h in history:
                out += '<|im_start|>{} {}<|im_end|>'.format(h['role'], h['content'])
            return out
    
    def _website_search(self, query):
        """
        使用爬虫的方式进行网页检索
        """
        print('搜索网页中......')
        try_num = 0

        while True:
            headers = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }
            url = 'https://cn.bing.com/academic/search?q={}'.format(query)
            r = requests.get(url, headers=headers)
            html = etree.HTML(r.text)
            out = ''
            for i, h in enumerate(html.xpath("//ol/li[@class='aca_algo']")):
                title = ''.join(h.xpath("h2//text()"))
                content = ''.join(h.xpath("div[@class='aca_caption']/div[@class='caption_abstract']//text()"))
                out += ('资料{}:'.format(i) + "(" + title + ")" + content)
            
            try_num += 1
            if len(out) != 0:
                return out
            
            elif try_num >= 10 and len(out)==0:
                return ''


    def stream_chat(self, query):
        if self.website_search:
            info = self._website_search(query)
            if len(info) != 0:
                query = '你可以参考的资料如下: ' + info + '请结合资料回答我的问题: ' + query

        self.history.append(
            {"role": "user", "content": query}
        )

        query = self._history_to_string(self.history)+ '<|im_start|>user ' + query + '<|im_end|> <|im_start|>assistant '
        inputs = self.tokenizer(query, return_tensors='pt').to('cuda:0')
        data = self.model.generate(**inputs, streamer=self.streamer, max_new_tokens=500)

        out = data.detach().to('cpu').numpy()[0]
        out = out[np.where(out==151644)[0][-1]:]
        out = self.tokenizer.decode(out).replace('<|im_start|>', '').replace('<|im_start|>', '').replace('<|im_end|>', '').replace('<|endoftext|>', '')
        self.history.append(
            {
                "role": "assistant", 
                "content": out[:-1][10:]
            }
        )
        return out[:-1][10:]
    
    def iter_chat(self, query):
        raise NotImplementedError

if __name__ == '__main__':
    m = myLLM(1, 2)
    m.set_system('你是一个金融助手')
    m.stream_chat('你好')
