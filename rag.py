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

# 4. 初始化 embedding 模型
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_class = HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_class = HuggingFaceEmbeddings

# 创建 embedding 模型
embed_model = embedding_class(
    model_name='maidalun1020/bce-embedding-base_v1',
    model_kwargs={'device': 'cuda:0'},
    encode_kwargs={'batch_size': 32, 'normalize_embeddings': True}
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

# 创建基础检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"score_threshold": 0.3, "k": 10}
)

# 8. 创建压缩检索器
compression_retriever = SimpleContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=reranker
)

# 9. 测试
if __name__=="__main__":
    response = compression_retriever.invoke("Dian团队是谁创立的?")
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
    