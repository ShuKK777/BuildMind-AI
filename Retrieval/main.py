import torch
import faiss
import pickle

import open_clip
import pandas as pd
import os.path as osp, os
from config import model_name, pretrained, feat_schema_path, image_root

## 创建服务
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from fastapi.staticfiles import StaticFiles

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 创建服务
app = FastAPI()


# 挂载图片文件
app.mount("/images", StaticFiles(directory=os.path.abspath(image_root)), name="images")


print(f">>>> 加载模型中...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)


@torch.no_grad()
def get_text_embed(text: str):
    text = tokenizer([text])
    text_features = model.encode_text(text, normalize=False)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.detach()


def load_faiss_index(index_path="image_index.faiss"):
    return faiss.read_index(index_path)


def load_image_paths(path="image_paths.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


## faiss -----------------------------
def search_by_faiss(text_query, image_paths, index, top_k=5):
    text_emb = get_text_embed(text_query).cpu().numpy().astype("float32")
    scores, indices = index.search(text_emb, min(top_k, len(image_paths)))
    semantic_images = [image_paths[i] for i in indices[0]]
    return semantic_images, scores[0].tolist()


## semantic --------------------------
def load_image_embed(save_path: str = "image_embed.pt"):
    image_embed = torch.load(save_path)
    return image_embed["image_paths"], image_embed["image_embed"]


@torch.no_grad()
def search_by_torch(text_query: str, image_paths, image_embed, top_k=5):
    text_features = get_text_embed(text_query)
    text_probs = (100.0 * text_features @ image_embed.T).softmax(dim=-1).cpu().squeeze()
    logits, img_ids = text_probs.sort(descending=True)
    logits = logits.tolist()[:top_k]
    semantic_images = [image_paths[i] for i in img_ids.tolist()[:top_k]]

    return semantic_images, logits


print(">>>> 加载图片索引中...")
# index, image_paths = load_faiss_index(), load_image_paths()
image_paths, image_embed = load_image_embed("image_embed.pt")

## 预处理feat
print(">>>> 读取图片信息...")
df = pd.read_excel(feat_schema_path)
columns = df.columns.tolist()


class SearchRequest(BaseModel):
    text_query: str
    top_k: int = 5
    filters: Optional[Dict[str, str]] = None  # 过滤条件


@app.post("/semantic_search")
def semantic_search(req: SearchRequest) -> list:
    # 参数处理
    text_query = req.text_query
    top_k = max(1, req.top_k)
    filters = req.filters or {}
    print(f"{text_query=}")
    print(f"{top_k=}")
    print(f"{filters=}")

    # 过滤图片
    tmp_df = df.copy()
    filters = {k: v for k, v in filters.items() if k in filters}
    for k, v in filters.items():
        tmp_df = tmp_df[tmp_df[k] == v]
    filter_img_names = tmp_df["img_id"].tolist()
    print(f"{filter_img_names=}")

    # semantic_images, scores = search_by_faiss(
    #     text_query, image_paths, index, top_k=100000
    # )
    semantic_images, scores = search_by_torch(
        text_query, image_paths, image_embed, top_k=100000
    )

    print(f"检索结果数: {len(semantic_images)}")
    print(list(zip(semantic_images, scores)))

    semantic_images = [
        [osp.basename(img_path), score]
        for img_path, score in zip(semantic_images, scores)
        if osp.basename(img_path).rsplit(".", 1)[0] in filter_img_names
    ]
    print(f"筛选结果数: {len(semantic_images)}")
    semantic_images = semantic_images[:top_k]

    print(f"最终检索结果数: {len(semantic_images)}")
    print(semantic_images)

    torch.cuda.empty_cache()

    return semantic_images


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8848)
