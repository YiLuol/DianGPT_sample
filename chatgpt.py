from dotenv import load_dotenv
import os
from openai import OpenAI
from trans import SystemMessage,HumanMessage,AIMessage
import json
from rag import compression_retriever
"""
PROXY_URL="http://172.17.0.1:1081"
os.environ["HTTP_PROXY"]=PROXY_URL
os.environ["HTTPS_PROXY"]=PROXY_URL
os.environ["http_proxy"]=PROXY_URL
os.environ["https_proxy"]=PROXY_URL
"""

load_dotenv()
api_key=os.getenv("DEEPSEEK_API_KEY")
base_url = "https://api.ppinfra.com/openai"
modelname = "deepseek/deepseek-v3.2-exp"

class Agent_Model:
    
    def __init__(self,api_key=api_key,system_prompt=None,try_load_history=False,model_name=modelname):
        self.model_name=model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,   
        )
        self.system_prompt=system_prompt
        self.default_file_path="chat_history.json"
        self.messages=[]
        if try_load_history:
            self.load_history()
        if self.system_prompt:
            self.messages.append({"role":"system","content":self.system_prompt})
            
    def send_message(self,input,temperature=0.1,max_completion_tokens=100,tools=None):             #一般形式的对话
        if type(input)==str:
            self.messages.append({"role":"user","content":input})
        elif type(input)==list:
            for index in input:
                self.messages.append({"role":"user","content":index})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=tools if tools !=None else None,
            tool_choice="auto" if tools !=None else "none"
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                if function_name=="retrieve_from_knowledge_base":
                    query = function_args.get("query")
                    function_response = compression_retriever.invoke(query)
                    doclist=[]
                    for i, doc in enumerate(function_response):
                        doclist.append(doc.page_content)
                self.messages.append(response_message)
                """
                self.messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
                """
                self.messages.append({"role":"user","content":f"检索工具返回的结果是：{function_response}"})
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                messages=self.messages
            )
            self.delete_tools()
            return final_response.choices[0].message.content
        else:
            return response_message.content
    
    def not_send_message(self,temperature=0.1,max_completion_tokens=100):               #不向message中插入对话但得到回答
        response = self.client.chat.completions.create(
            model=modelname,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            messages=self.messages
        )
        ai_response=response.choices[0].message.content
        self.messages.append({"role":"assistant","content":ai_response})
        return ai_response
    
    def clear_chat_history(self):
        self.messages=[]
        if self.system_prompt:
            self.messages.append({"role":"system","content":self.system_prompt})
            
    def invoke(self,history):                                                           #向message中插入对话但不得到回答
        self.messages+=history
        
    def manage_chat_history(self,max_round=3):
        max_length=2*max_round
        if self.messages[0]["role"]=="system" and len(self.messages)>max_length+1:
            temporary_messages=self.messages[-max_length:]
            self.messages=[self.messages[0]]+temporary_messages
        elif self.messages[0]["role"]!="system" and len(self.messages)>max_length:
            temporary_messages=self.messages[-max_length:]
            self.messages=temporary_messages
        return max_length
    
    def invoke_message(self,history):                                                   #直接向message中插入对话并得到回答
        self.invoke(history)
        return self.not_send_message()
    
    def save_history(self,change_file_path=False,file_path=""):
        if not change_file_path:
            file_path=self.default_file_path
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        print(f"历史已保存到：{file_path}")
        
    def load_history(self,change_file_path=False,file_path=""):
        """从JSON文件加载对话历史"""
        if not change_file_path:
            file_path=self.default_file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.messages = json.load(f)
            print(f"已加载历史：共{len(self.messages)}条消息")
        except FileNotFoundError:
            print(f"未找到历史文件：{file_path}")
    
    def delete_tools(self):
        del self.messages[-2]