import streamlit as st
from predict import predict_ticket

st.set_page_config(page_title="Ticket Classifier", layout="centered")

# --- App Logo (Replaces 🎫) ---
st.image("https://placehold.co/800x200/2980B9/FFFFFF?text=Support+Ticket+System", use_container_width=True)
st.title("IT Service Ticket Classifier")
st.write("Enter your issue below to classify the support ticket.")

# --- Input Icon (Replaces 📝) ---
st.image("https://placehold.co/200x50/34495E/FFFFFF?text=Ticket+Details", width=150)
user_input = st.text_area("Ticket Description", height=200, placeholder="Describe your issue here...")

if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter a ticket description")
    else:
        with st.spinner("Analyzing ticket..."):
            result = predict_ticket(user_input)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Classification Complete")
            
            # --- Image Assets (Replace these URLs with your own hosted images) ---
            category_images = {
                "Hardware": "https://placehold.co/600x200/2C3E50/FFFFFF?text=Hardware+Issue",
                "Access": "https://placehold.co/600x200/E74C3C/FFFFFF?text=Access+Denied",
                "Software": "https://placehold.co/600x200/8E44AD/FFFFFF?text=Software+Issue",
                "HR Support": "https://placehold.co/600x200/27AE60/FFFFFF?text=HR+Support",
                "Purchase": "https://placehold.co/600x200/F39C12/FFFFFF?text=Purchase+Request",
                "Administrative rights": "https://placehold.co/600x200/34495E/FFFFFF?text=Admin+Rights",
                "Internal Project": "https://placehold.co/600x200/16A085/FFFFFF?text=Internal+Project",
                "Miscellaneous": "https://placehold.co/600x200/95A5A6/FFFFFF?text=Miscellaneous"
            }
            
            priority_images = {
                "High": "https://placehold.co/600x200/C0392B/FFFFFF?text=High+Priority",
                "Medium": "https://placehold.co/600x200/D35400/FFFFFF?text=Medium+Priority",
                "Low": "https://placehold.co/600x200/27AE60/FFFFFF?text=Low+Priority"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category Icon (Replaces 📂)
                st.image("https://placehold.co/50x50/2980B9/FFFFFF?text=Cat", width=40)
                st.markdown("**Category**")
                cat = result['category']
                img_url = category_images.get(cat, category_images["Miscellaneous"])
                st.image(img_url, use_container_width=True)
                st.caption(f"Detected: {cat}")
            
            with col2:
                # Priority Icon (Replaces 🔥)
                st.image("https://placehold.co/50x50/C0392B/FFFFFF?text=imp", width=40)
                st.markdown("**Priority**")
                prio = result.get('priority', 'Low')
                prio_url = priority_images.get(prio, priority_images["Low"])
                st.image(prio_url, use_container_width=True)
                st.caption(f"Level: {prio}")

st.markdown("---")
st.caption("Powered by Scikit-Learn | IT Service Ticket Dataset")
