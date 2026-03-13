def SystemMessage(content):
    return {"role":"system","content":content}
def HumanMessage(content):
    return {"role":"user","content":content}
def AIMessage(content):
    return {"role":"assistant","content":content}
