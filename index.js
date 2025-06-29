require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const axios = require("axios");
const { v4: uuidv4 } = require("uuid");
const { InferenceClient } = require("@huggingface/inference");

const {
  ClerkExpressRequireAuth,
  ClerkExpressWithAuth,
} = require("@clerk/clerk-sdk-node");

const app = express();
const hfClient = new InferenceClient(process.env.HUGGINGFACE_API_KEY);

// ✅ Middleware
app.use(cors());
app.use(express.json());
app.use(ClerkExpressWithAuth()); // Clerk context injection

// ✅ Debug Clerk Keys
console.log("🔐 Clerk Secret Key Loaded:", !!process.env.CLERK_SECRET_KEY);

// ✅ Debug Request Token (for development only)
// app.use((req, res, next) => {
//   console.log("📩 Incoming Token:", req.headers.authorization || "None");
//   next();
// });

// ✅ MongoDB Connection
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB connected"))
  .catch((err) => console.error("❌ MongoDB error:", err));

// ✅ Schema
const ChatSchema = new mongoose.Schema({
  sessionId: String,
  userId: String,
  title: String,
  messages: Array,
  updatedAt: { type: Date, default: Date.now },
});

const Chat = mongoose.model("Chat", ChatSchema);

// ✅ AI Provider Handlers

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

async function fetchFromGroqWithRetry(model, messages, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await axios.post(
        "https://api.groq.com/openai/v1/chat/completions",
        { model, messages },
        {
          headers: {
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
            "Content-Type": "application/json",
          },
        }
      );
      return res;
    } catch (err) {
      const msg = err.response?.data?.message || "";
      if (i === retries - 1 || !msg.includes("no healthy upstream")) throw err;
      await delay(2000);
    }
  }
}

async function fetchFromHuggingFace(model, messages) {
  try {
    const hfResponse = await hfClient.chatCompletion({
      provider: model.startsWith("google/") ? "together" : "featherless-ai",
      model,
      messages,
      temperature: 0.7,
      max_tokens: 300,
    });
    return hfResponse.choices[0]?.message?.content || "No response generated.";
  } catch (err) {
    console.error("💥 Hugging Face error:", err);
    throw err;
  }
}

async function fetchFromOpenRouter(model, messages) {
  try {
    const res = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      { model, messages },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
          "HTTP-Referer": "http://localhost:3000",
          "X-Title": "AI Query Assistant",
        },
      }
    );
    return res.data.choices[0].message.content;
  } catch (err) {
    console.error("💥 OpenRouter error:", err.response?.data || err.message);
    throw err;
  }
}

// ✅ Unified Answer Generator
async function generateAnswer(question, messages, model, provider = "groq") {
  const trimmed = messages.slice(-8);
  let answer;

  if (provider === "huggingface") {
    answer = await fetchFromHuggingFace(model, trimmed);
  } else if (provider === "openrouter") {
    answer = await fetchFromOpenRouter(model, trimmed);
  } else {
    const response = await fetchFromGroqWithRetry(model, trimmed);
    answer = response.data.choices[0].message.content;
  }

  return answer
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(?!\s)(.+?)\*/g, "<em>$1</em>")
    .replace(/^\*\s(.+)/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/<\/li>\s*<li>/g, "</li><li>")
    .replace(/\n+/g, "<br>");
}

// ✅ Clerk-Protected Routes

app.post("/query", ClerkExpressRequireAuth(), async (req, res) => {
  const userId = req.auth.userId;
  const { sessionId, question, model, provider = "groq" } = req.body;

  try {
    let chat = await Chat.findOne({ sessionId, userId });
    if (!chat) {
      console.warn("🚫 Forbidden: Session not owned by user", {
        sessionId,
        userId,
      });
      return res.status(403).json({ error: "Unauthorized session." });
    }
    
    const messages = [
      { role: "system", content: "You are a helpful assistant." },
      ...chat.messages,
      { role: "user", content: question },
    ];

    const answer = await generateAnswer(question, messages, model, provider);

    chat.messages.push({ role: "user", content: question });
    chat.messages.push({ role: "assistant", content: answer });
    chat.updatedAt = new Date();

    if (!chat.title)
      chat.title =
        question.length > 40 ? question.slice(0, 40) + "..." : question;

    await chat.save();

    res.json({ answer });
  } catch (err) {
    console.error("❌ /query error:", err);
    res.status(500).json({ error: "AI query failed." });
  }
});

app.post("/regenerate", ClerkExpressRequireAuth(), async (req, res) => {
  const { sessionId, model, provider } = req.body;
  const userId = req.auth.userId;

  try {
    const chat = await Chat.findOne({ sessionId, userId });
    if (!chat || chat.messages.length === 0)
      return res.status(400).json({ error: "No chat history." });

    const lastUserMsg = [...chat.messages]
      .reverse()
      .find((m) => m.role === "user");
    if (!lastUserMsg)
      return res.status(400).json({ error: "No last user message." });

    const messages = [
      { role: "system", content: "You are a helpful assistant." },
      ...chat.messages.filter((m) => m.role !== "assistant"),
    ];

    const answer = await generateAnswer(
      lastUserMsg.content,
      messages,
      model,
      provider
    );
    chat.messages.push({ role: "assistant", content: answer });
    chat.updatedAt = new Date();
    await chat.save();

    res.json({ answer });
  } catch (err) {
    console.error("❌ /regenerate error:", err);
    res.status(500).json({ error: "Failed to regenerate." });
  }
});

app.post("/start-session", ClerkExpressRequireAuth(), async (req, res) => {
  const userId = req.auth.userId;
  const sessionId = uuidv4();
  await Chat.create({ sessionId, userId, messages: [] });
  res.json({ sessionId });
});

app.get("/history/:sessionId", ClerkExpressRequireAuth(), async (req, res) => {
  const userId = req.auth.userId;
  const { sessionId } = req.params;

  try {
    const chat = await Chat.findOne({ sessionId, userId });
    if (!chat) return res.json({ history: [] });

    const history = [];
    for (let i = 0; i < chat.messages.length; i++) {
      if (
        chat.messages[i].role === "user" &&
        chat.messages[i + 1]?.role === "assistant"
      ) {
        history.push({
          question: chat.messages[i].content,
          answer: chat.messages[i + 1].content,
        });
        i++;
      }
    }

    res.json({ history: history.reverse() });
  } catch (err) {
    console.error("❌ /history error:", err);
    res.status(500).json({ error: "Failed to fetch history." });
  }
});
app.get("/", (req, res) => {
  res.send("Backend is working!");
});


app.get("/sessions", ClerkExpressRequireAuth(), async (req, res) => {
  const userId = req.auth.userId;

  try {
    const sessions = await Chat.find(
      { userId },
      "sessionId title updatedAt"
    ).sort({ updatedAt: -1 });

    res.json(sessions);
  } catch (err) {
    console.error("❌ /sessions error:", err);
    res.status(500).json({ error: "Failed to fetch sessions." });
  }
});

app.delete(
  "/history/:sessionId",
  ClerkExpressRequireAuth(),
  async (req, res) => {
    const userId = req.auth.userId;
    const { sessionId } = req.params;

    await Chat.findOneAndDelete({ sessionId, userId });
    res.json({ message: "Deleted session." });
  }
);

app.patch(
  "/sessions/:sessionId",
  ClerkExpressRequireAuth(),
  async (req, res) => {
    const userId = req.auth.userId;
    const { sessionId } = req.params;
    const { title } = req.body;

    await Chat.findOneAndUpdate({ sessionId, userId }, { title });
    res.json({ message: "Renamed session." });
  }
);

app.post("/api/get-images", async (req, res) => {
  const { personNames } = req.body;
  if (!Array.isArray(personNames) || personNames.length === 0)
    return res.status(400).json({ error: "No person names provided." });

  const fetchImage = async (name) => {
    let fallback = null;
    try {
      const res = await axios.get(
        "https://www.googleapis.com/customsearch/v1",
        {
          params: {
            key: process.env.GOOGLE_API_KEY,
            cx: process.env.GOOGLE_CSE_ID,
            searchType: "image",
            q: name,
            num: 1,
          },
        }
      );
      const item = res.data.items?.[0];
      fallback = item?.link || item?.image?.thumbnailLink || null;
    } catch (err) {
      console.warn(`⚠️ Google failed for ${name}:`, err.message);
    }
    return { name, fallback };
  };

  const resultArray = await Promise.all(personNames.map(fetchImage));
  const results = Object.fromEntries(resultArray.map((r) => [r.name, r]));
  res.json(results);
});

app.get("/protected-route", ClerkExpressRequireAuth(), (req, res) => {
  res.json({
    message: "✅ Authenticated successfully!",
    userId: req.auth.userId,
  });
});

// ✅ Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
