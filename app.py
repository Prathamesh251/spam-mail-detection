import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Spam Mail Detection App")

st.write("Enter a message to check if it is Spam or Not Spam")

user_input = st.text_area("Enter Message")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        spam_prob = model.predict_proba(text_tfidf)[0][1]
        confidence = max(spam_prob, 1 - spam_prob)

        if 0.50 <= confidence <= 0.70:
            label = "Spam" if prediction == 1 else "Not Spam"
            st.warning(f"⚠️ Borderline / Uncertain — Likely {label} ({confidence:.1%} confidence). Please review manually.")
        elif prediction == 1:
            st.error(f"🚫 Spam Message ({confidence:.1%} confidence)")
        else:
            st.success(f"✅ Not Spam ({confidence:.1%} confidence)")
