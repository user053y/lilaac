from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from supabase import create_client, Client
import sys


# --- 2. TARIK DATA & LATIH AI SAAT SERVER MENYALA ---
print("Mengunduh data pengetahuan dari Supabase...")
try:
    response = supabase.table('dataset_bot').select('*').execute()
    dataset_db = response.data
    
    if not dataset_db:
        print("Peringatan: Tabel dataset_bot kosong. Tambahkan data di Supabase dulu.")
        sys.exit(1)
        
except Exception as e:
    print(f"Gagal terhubung ke Supabase: {e}")
    sys.exit(1)

kalimat = []
label = []
responses_dict = {}

# Menyusun data dari database ke dalam format yang dimengerti AI
for baris in dataset_db:
    tag = baris['tag']
    responses_dict[tag] = baris['responses']
    
    for keyword in baris['keywords']:
        kalimat.append(keyword)
        label.append(tag)

print("Melatih AI dengan data dari database...")
vectorizer = TfidfVectorizer(lowercase=True)
X_train = vectorizer.fit_transform(kalimat)
model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=42)
model.fit(X_train, label)
print("AI selesai dilatih dan API siap digunakan!")

# --- 3. KONFIGURASI SERVER FASTAPI ---
app = FastAPI()

# Mengizinkan frontend (HTML/JS) berkomunikasi dengan API ini
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    user_message: str

# --- 4. ENDPOINT UNTUK MENERIMA PESAN DARI WEB ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    input_text = request.user_message
    
    # AI mengubah teks pengguna menjadi angka dan memprediksi maknanya
    input_vector = vectorizer.transform([input_text])
    prediksi_tag = model.predict(input_vector)[0]
    
    # Mengambil balasan acak dari tag yang cocok
    if prediksi_tag in responses_dict:
        bot_reply = random.choice(responses_dict[prediksi_tag])
    else:
        bot_reply = "Maaf, aku belum ngerti maksudmu."
        
    # Menyimpan riwayat obrolan ke tabel chat_history di Supabase
    try:
        supabase.table('chat_history').insert({
            "user_message": input_text,
            "bot_response": bot_reply
        }).execute()
    except Exception as e:
        print(f"Gagal menyimpan riwayat ke DB: {e}")

    # Mengirimkan balasan kembali ke web HTML
    return {"reply": bot_reply}