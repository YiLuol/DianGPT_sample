from chatgpt import Agent_Model
from trans import SystemMessage
import json
from judge1 import judge
import pandas as pd
import re
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_knowledge_base",
            "description": "当用户的问题涉及到可能有关于Dian团队的信息或者私有知识库、内部文档、特定政策和未经训练的事实信息时，必须调用此工具来获取准确信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于在知识库中检索的关键词或问题。"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

chat_model=Agent_Model()

Instruction=[SystemMessage("""
                           你是Dian团队开发的助手，旨在解答同学对Dian团队的一系列问题，并尽量不要回答与Dian团队无关的问题。请尽可能地使用工具提供的知识库来提供更真实的信息。知识库中包含过往的问答案例，每个案例附有总评分（total_score）（满分10分）。请优先参考总评分高的案例的回答风格和逻辑，避免重复低评分案例中的错误。如果检索到的案例与当前问题高度相关，可以借鉴其回答方式。
                           重点：请不要直接用提供的信息回答，需要转化为用户能看懂的语言，即具有完整的语句和规范的表达。回答需简洁易懂。
                           """)]

def chat():
    chat_model.invoke(Instruction)
    learn_mod=0
    print(f"是否启动学习模式？(y/n)")
    while(1):
        a=input()
        if a=="y":
            print(f"已开启!")
            learn_mod=1
            break
        elif a=="n":
            print(f"未开启!")
            break
    print(f"AI:你好！我是DianGPT，有什么需要帮助的吗？")
    while(1):
        chat_model.manage_chat_history()
        user_input=input("你：")
        if user_input.lower()=="退出":
            print("AI:再见！")
            break
        response=chat_model.send_message(user_input,tools=tools)
        print(f"AI：{response}\n")
        if learn_mod==1:
            path="eval.xlsx"
            data=pd.read_excel(path)
            score=[]
            while(len(score)<6):
                score=re.findall(r"\d+",judge(chat_model.messages[-2:-1]))
            try:
                partial_score,total_score=score[0:5],score[5]
                new_row={'input':user_input,'response':response,'partial_score':partial_score,'total_score':total_score}
                data.loc[len(data)] = new_row
                # 逐步保存结果
                try:
                    data.to_excel(path, index=False)
                except Exception as e:
                    print(e)
                    print(f"Failed to save data at row")
                    continue
            except IndexError as e:
                print(e)
                print(f"failed to eval this turn.")
                continue
            
            
if __name__=="__main__":
    while(1):
        try:
            chat()
            break
        except SystemError as a:
            print(a)
            continue
    #Dian团队是谁创立的?
    #Dian团队有什么成就?