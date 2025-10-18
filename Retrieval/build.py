import os
import pickle
import torch
import faiss
import open_clip
from tqdm import tqdm
from PIL import Image
from config import model_name, pretrained, image_root

print(f">>> load model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)


@torch.no_grad()
def create_image_embed(image_paths: str | list[str], save_path="image_embed.pt"):
    image_paths = image_paths if isinstance(image_paths, list) else []
    images_embed = []
    for img_path in tqdm(image_paths):
        image = preprocess(Image.open(img_path)).unsqueeze(0)
        image_features = model.encode_image(image, normalize=False)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        images_embed.append(image_features.cpu())
        torch.cuda.empty_cache()

    if len(images_embed) == 0:
        torch.save({"image_paths": [], "image_embed": []}, save_path)
        return

    images_embed = torch.cat(images_embed)
    print(images_embed.shape)
    torch.save({"image_paths": image_paths, "image_embed": images_embed}, save_path)

    return images_paths, images_embed


def build_faiss_index(image_embeddings: torch.Tensor):
    dim = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积近似于余弦相似度（embedding已归一化）
    faiss_embeddings = image_embeddings.numpy().astype("float32")
    index.add(faiss_embeddings)
    return index


def save_faiss_index(index, index_path="image_index.faiss"):
    faiss.write_index(index, index_path)


def save_image_paths(paths, path="image_paths.pkl"):
    with open(path, "wb") as f:
        pickle.dump(paths, f)


images_paths = [
    image_root + i for i in os.listdir(image_root) if os.path.isfile(image_root + i)
]
images_paths, images_embed = create_image_embed(images_paths)

faiss_index = build_faiss_index(images_embed)
save_faiss_index(faiss_index)
save_image_paths(images_paths)
