# Bajaj HackRx 6.0
# Team Name: Nammadha

---

## 📦 API Endpoint

### POST /hackrx/run

**Headers**
Authorization: Bearer <YOUR_API_KEY>  
Content-Type: application/json

**Request Body**
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the policy coverage amount?",
    "Are dental treatments included?"
  ]
}

**Response**
{
  "answers": [
    "The policy coverage amount is ₹5,00,000.",
    "Yes, dental treatments are covered under outpatient benefits."
  ]
}

---

## 🔑 Authentication

This API uses bearer token-based access. Set your API key in a `.env` file:

TEAM_API_KEY=your-secret-api-key

The Authorization header must be included in each request:

Authorization: Bearer your-secret-api-key

---

## 📁 Project Structure

.
├── agent.py          # Logic to process documents and generate answers  
├── main.py           # FastAPI app and routes  
├── models.py         # Pydantic request models  
├── .env              # Environment variables (TEAM_API_KEY, GEMINI_API_KEY)  
├── requirements.txt  # Python dependencies

---

## 🚀 Running Locally

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload

Then, go to terminal and run:
`Invoke-WebRequest -Uri 'http://127.0.0.1:8000/hackrx/run' -Method POST -Headers @{ "Content-Type" = "application/json"; "Authorization" = <TEAM_API_KEY> } -Body '{"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D", "questions": ["What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?", "What is the waiting period for cataract surgery?"]}'`
or,
`POST /hackrx/run Content-Type: application/json Accept: application/json Authorization: Bearer <TEAM_API_KEY> { "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D", "questions": ["What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?", "What is the waiting period for pre-existing diseases (PED) to be covered?", "Does this policy cover maternity expenses, and what are the conditions?", "What is the waiting period for cataract surgery?", "Are the medical expenses for an organ donor covered under this policy?", "What is the No Claim Discount (NCD) offered in this policy?", "Is there a benefit for preventive health check-ups?", "How does the policy define a 'Hospital'?", "What is the extent of coverage for AYUSH treatments?", "Are there any sub-limits on room rent and ICU charges for Plan A?"]}`

---

## 🤝 Built For

This project is developed as part of the HackRx Bajaj Hackathon challenge to enhance document comprehension using AI.
