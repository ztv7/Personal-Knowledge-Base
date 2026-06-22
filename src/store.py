#https://pydantic.com.cn/
import chromadb
from pydantic import BaseModel,Field
from chromadb.utils import embedding_functions
from chromadb import Collection
from typing import List,Dict,Any,Optional
from src.config import SILICONFLOW_API_KEY,SILICONFLOW_BASE_URL,SILICONFLOW_EMBEDDING_MODEL,CHROMADB_CLIENT_PATH

class Database(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    client: Optional[chromadb.PersistentClient] = Field(default=None, init=False)
    collection: Optional[Collection] = Field(default=None, init=False)
    # 新增：存储所有已入库完整文档名列表
    doc_name_map: Dict[str, str] = Field(default_factory=dict, init=False)

    def model_post_init(self,__context:Any):
        self.client = chromadb.PersistentClient(path=CHROMADB_CLIENT_PATH)
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_base=SILICONFLOW_BASE_URL,
            api_key=SILICONFLOW_API_KEY,
            model_name=SILICONFLOW_EMBEDDING_MODEL,
        )
        self.collection = self.client.get_or_create_collection(name='collection1',embedding_function=ef)
        # 启动时读取库中所有文档source，回填映射字典
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        for meta in all_meta:
            if meta and "source" in meta:
                full_name = meta["source"]
                self.doc_name_map[full_name] = full_name

    def add(self,ids:List[str],documents:List[str],metadatas:List[Dict[str,Any]]):
        self.collection.upsert(ids=ids,documents=documents,metadatas=metadatas)
        # 新增入库时自动存入文档名映射
        for meta in metadatas:
            full_source = meta["source"]
            self.doc_name_map[full_source] = full_source

    def query_question(self,query_text:str,where:dict=None,n_result=5)->Dict[str,Any]:
        kwargs = {"query_texts":[query_text],"n_results":n_result}
        if where and where != {}:
            kwargs["where"] = where
        return self.collection.query(**kwargs)
    
    def close(self):
        self.collection = None
        self.client = None