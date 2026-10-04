import os
import random
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Pastikan SUPABASE_URL dan SUPABASE_KEY sudah diisi di Vercel!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Variabel Global Otak AI
kalimat, label, responses_dict = [], [], {}
vectorizer = TfidfVectorizer(lowercase=True)
model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)

def load_and_train():
    """Fungsi untuk menarik data dari DB dan melatih AI (Bisa dipanggil kapan saja)"""
    global kalimat, label, responses_dict, vectorizer, model
    response = supabase.table('dataset_bot').select('*').execute()
    dataset_db = response.data
    
    kalimat.clear()
    label.clear()
    responses_dict.clear()
    
    for baris in dataset_db:
        tag = baris['tag']
        responses_dict[tag] = baris['responses']
        for keyword in baris['keywords']:
            kalimat.append(keyword)
            label.append(tag)

    if kalimat:
        X_train = vectorizer.fit_transform(kalimat)
        model.fit(X_train, label)

# Latih AI saat server pertama menyala
load_and_train()

class ChatRequest(BaseModel):
    user_message: str

class TeachRequest(BaseModel):
    question: str
    answer: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    input_text = request.user_message.strip()
    is_fallback = False
    
    try:
        input_vector = vectorizer.transform([input_text])
        # AI mengukur persentase keyakinannya
        probs = model.predict_proba(input_vector)[0]
        max_prob = max(probs)
        prediksi_tag = model.classes_[probs.argmax()]
        
        # Jika keyakinan di bawah 30%, AI mengaku tidak tahu
        if max_prob < 0.3:
            bot_reply = "Maaf, aku belum ngerti maksudmu."
            is_fallback = True
        else:
            bot_reply = random.choice(responses_dict[prediksi_tag])
            
    except Exception as e:
        bot_reply = "Maaf, sistemku sedang bingung."
        is_fallback = True

    return {"reply": bot_reply, "is_fallback": is_fallback}

@app.post("/teach")
async def teach_endpoint(request: TeachRequest):
    """Endpoint baru untuk menerima pelajaran dari Pop-Up HTML"""
    new_tag = f"learned_{uuid.uuid4().hex[:6]}" # Buat tag acak unik
    
    try:
        # 1. Simpan ilmu baru ke Supabase
        supabase.table('dataset_bot').insert({
            "tag": new_tag,
            "keywords": [request.question.lower()],
            "responses": [request.answer]
        }).execute()
        
        # 2. Latih ulang otak AI detik itu juga!
        load_and_train()
        
        return {"status": "success", "message": "Terima kasih! Aku sudah bertambah pintar."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
