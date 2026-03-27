import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


# 1. 定义 BaseDocumentCompressor 基类
class BaseDocumentCompressor:#压缩文档
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        """压缩文档的方法"""
        raise NotImplementedError("子类必须实现此方法")

# 2. 使用本地模型的版本
class LocalBCEReranker(BaseDocumentCompressor):
    def __init__(self, model_name: str = 'maidalun1020/bce-reranker-base_v1',
                 top_n: int = 5, device: str = 'cuda:0', max_length: int = 504):
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self.max_length = max_length
        
        # 尝试加载本地模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if device.startswith('cuda'):
                self.model = self.model.to(device)
            self.model.eval()
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            self.tokenizer = None
            self.model = None
    
    def compute_score(self, pairs):
        """计算query-document对的分数"""
        if self.model is None or self.tokenizer is None:
            # 返回模拟分数
            return [0.5] * len(pairs)
        
        scores = []
        with torch.no_grad():
            for query, document in pairs:
                inputs = self.tokenizer(
                    query, 
                    document, 
                    truncation=True, 
                    max_length=self.max_length,
                    padding=True,
                    return_tensors='pt'
                )
                
                if self.device.startswith('cuda'):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits).item()
                scores.append(score)
        
        return scores
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,
    ) -> List[Document]:
        if not self.model:
            print("模型未加载，返回原始文档")
            return documents[:self.top_n]
        
        texts = [doc.page_content for doc in documents]
        pairs = [[query, text] for text in texts]
        
        scores = self.compute_score(pairs)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:self.top_n]]

# 3. 创建简单的 ContextualCompressionRetriever（不使用 Pydantic）
class SimpleContextualCompressionRetriever:
    """简单的压缩检索器，不继承 BaseRetriever 避免 Pydantic 问题"""
    def __init__(self, base_retriever, base_compressor):
        # 直接设置属性，不使用 Pydantic
        self.base_retriever = base_retriever
        self.base_compressor = base_compressor
    
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """获取相关文档并压缩"""
        # 第一步：使用基础检索器获取文档
        
        # 新版 LangChain 使用 invoke 方法而不是 get_relevant_documents
        try:
            # 尝试使用 invoke 方法
            documents = self.base_retriever.invoke(query, **kwargs)
        except AttributeError:
            # 如果 invoke 不存在，尝试使用 get_relevant_documents
            try:
                documents = self.base_retriever.get_relevant_documents(query, **kwargs)
            except AttributeError:
                # 如果都没有，尝试使用 _get_relevant_documents
                documents = self.base_retriever._get_relevant_documents(query, **kwargs)
        
        
        # 第二步：使用压缩器压缩文档
        if documents and self.base_compressor:
            compressed_docs = self.base_compressor.compress_documents(documents, query)
            return compressed_docs
        return documents
    
    def invoke(self, query: str, **kwargs) -> List[Document]:
        """新版本的调用方法"""
        return self.get_relevant_documents(query, **kwargs)
    
class MergedRetriever:
    """手动合并多个检索器，支持权重和去重（普通类，不继承 BaseRetriever）"""
    
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        if weights is None:
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            self.weights = weights
    
    def invoke(self, query: str, **kwargs):
        all_docs = []
        for retriever, weight in zip(self.retrievers, self.weights):
            # 兼容不同的调用方式
            try:
                docs = retriever.invoke(query, **kwargs)
            except AttributeError:
                docs = retriever.get_relevant_documents(query, **kwargs)
            for doc in docs:
                doc.metadata['retriever_weight'] = weight
            all_docs.extend(docs)
        
        # 去重（基于内容）
        unique = {}
        for doc in all_docs:
            key = doc.page_content
            if key not in unique:
                unique[key] = doc
        return list(unique.values())
    
    def get_relevant_documents(self, query: str, **kwargs):
        return self.invoke(query, **kwargs)
    
def build_excel_vectorstore(excel_path: str, embed_model, save_path: str = "./excel_faiss"):
    import pandas as pd
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    df = pd.read_excel(excel_path, engine='openpyxl')
    print("实际列名:", df.columns.tolist())

    if 'input' not in df.columns or 'response' not in df.columns:
        raise ValueError("Excel 必须包含 'input' 和 'response' 列")

    documents = []
    for idx, row in df.iterrows():
        question = str(row['input']) if pd.notna(row['input']) else ""
        answer = str(row['response']) if pd.notna(row['response']) else ""
        if not question and not answer:
            continue
        content = f"问题：{question}\n回答：{answer}"
        score = row.get('total_score')
        if pd.isna(score):
            score = row.get('partial_score')
        if pd.isna(score):
            score = None
        metadata = {
            "score": score,
            "row_idx": idx,
            "source": "excel",
            "input": question,
            "response": answer
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    if not documents:
        raise ValueError("没有有效的文档行，请检查 Excel 数据")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    # 过滤空文档
    docs = [doc for doc in docs if doc.page_content.strip()]
    # 手动截断超长文本（可选）
    max_len = 512
    for doc in docs:
        if len(doc.page_content) > max_len:
            doc.page_content = doc.page_content[:max_len]

    vectorstore = FAISS.from_documents(
        docs,
        embed_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    vectorstore.save_local(save_path)
    print(f"Excel 向量库已构建，共 {len(docs)} 个片段，保存至 {save_path}")
    return vectorstore



# 4. 初始化 embedding 模型
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_class = HuggingFaceEmbeddings

# 创建 embedding 模型
embed_model = HuggingFaceEmbeddings(
    model_name='maidalun1020/bce-embedding-base_v1',
    model_kwargs={'device': 'cuda:0'},
    encode_kwargs={
        'batch_size': 32,
        'normalize_embeddings': True
        # 移除 padding, truncation, max_length
    }
)

# 5. 初始化本地模型的 reranker
reranker = LocalBCEReranker(
    model_name='maidalun1020/bce-reranker-base_v1',
    top_n=5,
    device='cuda:0',
    max_length=504
)

# 6. 加载文档
documents = PyPDFLoader(r"data/1009-点亮人生.pdf").load()+Docx2txtLoader(r"data/2022团队年鉴.docx").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 7. 创建向量库和检索器
vectorstore = FAISS.from_documents(
    texts,
    embed_model,
    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
)
excel_vectorstore = build_excel_vectorstore(
    excel_path="eval.xlsx",  # 替换为您的 Excel 路径
    embed_model=embed_model,
    save_path="./excel_faiss"
)

# 创建基础检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"score_threshold": 0.3, "k": 10}
)
excel_retriever = excel_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"score_threshold": 0.3, "k": 10}  # 可根据需要调整
)
merged_retriever = MergedRetriever(
    retrievers=[retriever, excel_retriever],
    weights=[0.5, 0.5]
)

# 8. 创建压缩检索器
compression_retriever1 = SimpleContextualCompressionRetriever(
    base_retriever=merged_retriever,
    base_compressor=reranker
)
compression_retriever2 = SimpleContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=reranker
)

# 9. 测试
if __name__=="__main__":
    response = compression_retriever2.invoke("Dian团队是谁创立的?")
    doclist=[]
    #for i, doc in enumerate(response):
    #    doclist.append(doc.page_content)
    
    for i, doc in enumerate(response):
        print(f"\n文档 {i+1}:")
        content_preview = doc.page_content[:200].replace('\n', ' ')
        print(f"内容预览: {content_preview}...")
        if doc.metadata:
            print(f"元数据: {doc.metadata}")
        print("-" * 50)
    query = "Dian团队是哪个人创立的？"  # 选择一个与 Excel 中问题相似的问题
    results = compression_retriever1.invoke(query)

    print(f"共检索到 {len(results)} 个相关片段\n")
    for i, doc in enumerate(results):
        print(f"=== 片段 {i+1} ===")
        print(f"内容: {doc.page_content[:300]}...")
        if 'score' in doc.metadata:
            print(f"总评分: {doc.metadata['score']}")
        print()
