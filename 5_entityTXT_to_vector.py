with open("./entity2id.txt","r",encoding="utf-8") as f:
    entities = f.readlines()
    entities = [entity.strip().split()[0] for entity in entities]


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./distiluse-base-multilingual-cased-v1')


# encode entities
embeddings = model.encode(entities, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entities,
    "embeddings": embeddings,
}
import pickle
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

