# breast_ultrasound_ai_tutor_enhanced.py
import os
import io
from dotenv import load_dotenv

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34
from torchcam.methods import SmoothGradCAMpp

# Optional Groq tutor
try:
    from groq import Groq
except Exception:
    Groq = None

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

groq_client = None
if Groq is not None and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None

# ---------------------------
# Model definition
# ---------------------------
CLASS_NAMES = ["normal", "benign", "malignant"]

class YOLOMultitask(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = resnet34(weights="IMAGENET1K_V1")
        layers = list(base.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        cls_out = self.fc(self.avgpool(feat).view(feat.size(0), -1))
        seg_out = self.seg_head(feat)
        return cls_out, seg_out, feat

# ---------------------------
# Helpers
# ---------------------------
def blend_overlay(base_img, overlay_img, alpha=0.5):
    return (alpha * overlay_img + (1 - alpha) * base_img).astype(np.uint8)

def dice_score(y_true, y_pred, eps=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)

def compute_mask_metrics(mask):
    h, w = mask.shape
    area = mask.sum() / (h * w)
    ys, xs = np.where(mask > 0.5)
    centroid = (int(xs.mean()), int(ys.mean())) if len(xs) > 0 else (None, None)
    return {"area_fraction": float(area), "centroid": centroid, "height": h, "width": w}

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "yolo_multitask_busi1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
cam_extractor = None

if os.path.exists(MODEL_PATH):
    try:
        model = YOLOMultitask(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        cam_extractor = SmoothGradCAMpp(model.encoder)
    except Exception as e:
        st.warning(f"Model failed to load: {e}")
        model = None
else:
    st.warning(f"Model file not found at {MODEL_PATH}. Place your model there.")
    model = None

# ---------------------------
# Preprocessing
# ---------------------------
IMG_SIZE = (512, 512)
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# ---------------------------
# ---------------------------
st.set_page_config(
    page_title="AI Radiology Practice - Breast Ultrasound",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .step-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .instruction-box {
        background: #eff6ff;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #bfdbfe;
        margin: 0.5rem 0;
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 3px solid #10b981;
    }
    .warning-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #f59e0b;
    }
    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #10b981;
    }
    .reference-section {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🩺 AI Radiology Practice Platform</h1>
    <h3>Breast Ultrasound Interpretation Training</h3>
    <p>Practice identifying and classifying breast lesions with AI-powered feedback from Dr. Nova, your virtual radiology attending</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Session state
# ---------------------------
if "student_mask" not in st.session_state: st.session_state.student_mask = None
if "student_class" not in st.session_state: st.session_state.student_class = None
if "student_reasoning" not in st.session_state: st.session_state.student_reasoning = ""
if "tutor_messages" not in st.session_state: st.session_state.tutor_messages = []
if "revealed" not in st.session_state: st.session_state.revealed = False
if "ai_results" not in st.session_state: st.session_state.ai_results = None
if "selected_demo" not in st.session_state: st.session_state.selected_demo = None

# ---------------------------
# ---------------------------
with st.sidebar:
    st.markdown("### 📚 Learning Objectives")
    st.markdown("""
    <div class="reference-section">
    <b>By the end of this session, you should be able to:</b>
    <ul>
        <li>Identify suspicious lesions on breast ultrasound</li>
        <li>Differentiate between benign and malignant features</li>
        <li>Apply BI-RADS classification criteria</li>
        <li>Develop systematic image interpretation skills</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 BI-RADS Quick Reference")
    with st.expander("View BI-RADS Categories"):
        st.markdown("""
        **BI-RADS 1:** Negative (Normal)
        **BI-RADS 2:** Benign finding
        **BI-RADS 3:** Probably benign (<2% malignancy)
        **BI-RADS 4:** Suspicious (2-95% malignancy)
        **BI-RADS 5:** Highly suggestive of malignancy (>95%)
        """)
    
    st.markdown("### 🔍 Key Ultrasound Features")
    with st.expander("Benign vs Malignant Features"):
        st.markdown("""
        **Benign Features:**
        - Round or oval shape
        - Smooth, well-defined margins
        - Hyperechoic or isoechoic
        - Posterior enhancement
        - Wider than tall orientation
        
        **Malignant Features:**
        - Irregular shape
        - Spiculated or microlobulated margins
        - Hypoechoic or markedly hypoechoic
        - Posterior acoustic shadowing
        - Taller than wide orientation
        - Calcifications
        """)
    
    st.markdown("---")
    
    # Demo image selection
    st.markdown("### 🖼️ Select Case")
    demo_folder = "demo_ultrasounds"
    demo_images = [f for f in os.listdir(demo_folder) if f.endswith((".png",".jpg"))][:5]
    demo_dict = {f"Case {i+1}": os.path.join(demo_folder,f) for i,f in enumerate(demo_images)}
    
    selected_demo = st.selectbox("Available Cases", options=list(demo_dict.keys()), key="demo_selector")
    st.session_state.selected_demo = selected_demo
    
    st.markdown("---")
    
    st.markdown("### 🎯 Your Assessment")
    st.markdown('<div class="instruction-box"><b>Step 2:</b> After drawing your region of interest, select your classification below</div>', unsafe_allow_html=True)
    
    student_class = st.radio(
        "Classification",
        options=CLASS_NAMES,
        format_func=lambda x: {
            "normal": "Normal (No lesion)",
            "benign": "Benign (BI-RADS 2-3)",
            "malignant": "Malignant (BI-RADS 4-5)"
        }[x],
        index=0,
        key="class_selector"
    )
    st.session_state.student_class = student_class
    
    st.markdown("### 📝 Clinical Reasoning")
    student_reasoning = st.text_area(
        "Document your findings and reasoning:",
        placeholder="Describe the lesion characteristics:\n- Shape and margins\n- Echogenicity\n- Posterior features\n- Orientation\n- Your differential diagnosis",
        height=150,
        key="reasoning_input"
    )
    st.session_state.student_reasoning = student_reasoning

# ---------------------------
# Main content area
# ---------------------------
if st.session_state.selected_demo:
    image_path = demo_dict[st.session_state.selected_demo]
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    W, H = IMG_SIZE

    # ---------------------------
    # ---------------------------
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("### 📍 Step 1: Identify the Region of Interest")
    st.markdown("""
    <div class="instruction-box">
    <b>Instructions:</b>
    <ul>
        <li>Carefully examine the ultrasound image below</li>
        <li>Draw a bounding box around any suspicious lesion or area of concern</li>
        <li>Try to include the entire lesion with minimal surrounding tissue</li>
        <li>Consider: Is there a discrete mass? Are the margins well-defined?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 50, 50, 0.25)",
        stroke_width=2,
        stroke_color="#ef4444",
        background_image=image,
        update_streamlit=True,
        height=H,
        width=W,
        drawing_mode="rect",
        key="student_canvas",
    )

    student_mask = np.zeros(IMG_SIZE, dtype=np.float32)
    if canvas_result and canvas_result.json_data is not None:
        for shape in canvas_result.json_data["objects"]:
            left = int(shape["left"])
            top = int(shape["top"])
            width = int(shape["width"])
            height = int(shape["height"])
            student_mask[top:top+height, left:left+width] = 1.0
        st.session_state.student_mask = student_mask

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        if st.button("✅ Lock My Assessment", type="primary", use_container_width=True):
            if st.session_state.student_mask is None or st.session_state.student_mask.sum() == 0:
                st.markdown('<div class="warning-box">⚠️ Please draw a bounding box around the region of interest first!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Assessment locked! You can now consult with Dr. Nova or reveal AI analysis.</div>', unsafe_allow_html=True)
    
    with col_b:
        if st.button("🔄 Clear & Start Over", use_container_width=True):
            st.session_state.student_mask = None
            st.session_state.revealed = False
            st.session_state.ai_results = None
            st.rerun()
    
    with col_c:
        if st.button("➡️ Next Case", use_container_width=True):
            # Reset for new case
            st.session_state.student_mask = None
            st.session_state.revealed = False
            st.session_state.ai_results = None
            st.session_state.tutor_messages = []
            st.session_state.student_reasoning = ""
            current_idx = list(demo_dict.keys()).index(st.session_state.selected_demo)
            next_idx = (current_idx + 1) % len(demo_dict)
            st.session_state.selected_demo = list(demo_dict.keys())[next_idx]
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # ---------------------------
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("### 💬 Step 2: Consult with Dr. Nova (Your AI Attending)")
    st.markdown("""
    <div class="instruction-box">
    <b>How to use Dr. Nova:</b>
    <ul>
        <li><b>Before revealing AI results:</b> Ask for guidance on your interpretation, discuss differential diagnoses, or clarify ultrasound features</li>
        <li><b>General questions:</b> "What are the key features of a malignant breast lesion?" or "Explain posterior acoustic shadowing"</li>
        <li><b>Case-specific:</b> "Does my segmentation look accurate?" or "Should I be concerned about the margins?"</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    for msg in st.session_state.tutor_messages:
        avatar = "👨‍⚕️" if msg["role"] == "assistant" else "👨‍🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    chat_text = st.chat_input("Ask Dr. Nova about ultrasound features, your assessment, or radiology concepts...")
    if chat_text:
        st.session_state.tutor_messages.append({"role": "user", "content": chat_text})
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(chat_text)

        general_keywords = ["what is", "what are", "tell me about", "explain", "describe", "symptoms", "features", "signs", "risk", "causes", "treatment", "diagnosis", "bi-rads", "ultrasound"]
        is_general = any(keyword in chat_text.lower() for keyword in general_keywords)
        
        if is_general:
            tutor_context = f"""You are Dr. Nova, an experienced breast radiologist and medical educator at a teaching hospital.

A radiology resident has asked you a general question about breast imaging or pathology:
"{chat_text}"

Provide a clear, educational response that:
1. Directly answers their question with accurate medical information
2. Relates concepts to ultrasound imaging when relevant
3. Uses appropriate medical terminology but explains complex terms
4. Includes practical clinical pearls or teaching points
5. Keeps the response focused and concise (3-4 paragraphs maximum)

Remember: You are teaching a medical student who is learning breast radiology. Be supportive, clear, and educational."""

        else:
            if st.session_state.student_mask is not None and st.session_state.student_mask.sum() > 0:
                metrics = compute_mask_metrics(st.session_state.student_mask)
                tutor_context = f"""You are Dr. Nova, an experienced breast radiologist reviewing a case with a radiology resident.

CASE CONTEXT:
- Student has drawn a region of interest covering {metrics['area_fraction']*100:.1f}% of the image
- Student's classification: {st.session_state.student_class}
- Student's reasoning: {st.session_state.student_reasoning if st.session_state.student_reasoning else "Not provided"}

STUDENT QUESTION: "{chat_text}"

Provide focused feedback as an attending radiologist would:
1. Address their specific question directly
2. Comment on their segmentation approach (coverage, boundaries)
3. Evaluate their classification reasoning
4. Guide them toward key ultrasound features they should consider
5. Ask probing questions to develop their diagnostic reasoning
6. Provide specific, actionable suggestions for improvement

Keep your response conversational and educational, as if you're at the workstation together reviewing the case. Focus on teaching radiology interpretation skills."""

            else:
                tutor_context = f"""You are Dr. Nova, a breast radiologist teaching a resident.

The student asked: "{chat_text}"

They haven't drawn their region of interest yet. Encourage them to:
1. Carefully examine the entire ultrasound image
2. Look for any discrete masses or suspicious areas
3. Consider the echogenicity, margins, and posterior features
4. Draw a bounding box around any concerning findings

Then address their question briefly, reminding them to complete their assessment first."""

        assistant_reply = """**Dr. Nova's Guidance** (Tutor AI unavailable - Groq not configured)

**General Approach to Breast Ultrasound:**
- **Systematic Review:** Scan the entire image systematically
- **Identify Lesions:** Look for discrete masses with different echogenicity
- **Assess Margins:** Smooth margins suggest benign; irregular/spiculated suggest malignant
- **Posterior Features:** Enhancement (benign) vs shadowing (malignant)
- **Orientation:** Wider-than-tall (benign) vs taller-than-wide (malignant)

**Classification Tips:**
- **Normal:** No discrete lesion, normal breast parenchyma
- **Benign:** Well-defined, oval, hyperechoic, posterior enhancement
- **Malignant:** Irregular shape, hypoechoic, spiculated margins, posterior shadowing, microcalcifications

**Next Steps:** Complete your segmentation, document your reasoning, then we can discuss your findings in detail."""

        if groq_client:
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are Dr. Nova, an expert breast radiologist and dedicated medical educator. You provide clear, accurate, and supportive teaching to radiology residents."},
                        {"role": "user", "content": tutor_context}
                    ],
                    temperature=0.4,
                    max_tokens=600
                )
                assistant_reply = resp.choices[0].message.content
            except Exception as e:
                assistant_reply += f"\n\n*(Error connecting to AI tutor: {str(e)})*"

        st.session_state.tutor_messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant", avatar="👨‍⚕️"):
            st.markdown(assistant_reply)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # ---------------------------
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 Step 3: Compare with AI Analysis")
    st.markdown("""
    <div class="instruction-box">
    <b>Ready to see how you did?</b> The AI will show you its segmentation, classification, and attention maps (Grad-CAM).
    Use this to learn from differences and improve your interpretation skills.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Reveal AI Analysis & Compare", type="primary", use_container_width=True):
        if st.session_state.student_mask is None or st.session_state.student_mask.sum() == 0:
            st.markdown('<div class="warning-box">⚠️ Please complete your assessment (draw bounding box and select classification) before revealing AI results.</div>', unsafe_allow_html=True)
        elif model is None:
            st.error("❌ AI model not loaded. Please ensure the model file is in the correct location.")
        else:
            with st.spinner("🔄 AI is analyzing the ultrasound image..."):
                try:
                    img_tensor = preprocess(image).unsqueeze(0).to(device)
                    img_tensor.requires_grad = True
                    model.zero_grad()
                    cls_out, seg_out, feat = model(img_tensor)

                    probs = F.softmax(cls_out, dim=1).detach().cpu().numpy()[0]
                    ai_class_idx = int(torch.argmax(cls_out, dim=1).item())
                    ai_class = CLASS_NAMES[ai_class_idx]

                    seg_mask = seg_out.squeeze().cpu().detach().numpy()
                    seg_mask_bin = (seg_mask > 0.5).astype(np.float32)
                    seg_mask_full = np.array(Image.fromarray((seg_mask_bin * 255).astype(np.uint8)).resize(IMG_SIZE)) / 255.0

                    student_mask_resized = st.session_state.student_mask
                    dice = dice_score(seg_mask_full, student_mask_resized)

                    base_arr = np.array(image)
                    overlay_student = blend_overlay(base_arr, np.stack([student_mask_resized * 255] * 3, axis=-1).astype(np.uint8), alpha=0.4)
                    overlay_ai = blend_overlay(base_arr, np.stack([seg_mask_full * 255] * 3, axis=-1).astype(np.uint8), alpha=0.4)

                    # Grad-CAM visualization
                    cams_all = []
                    for i in range(len(CLASS_NAMES)):
                        model.zero_grad()
                        cls_score = F.softmax(cls_out, dim=1)[0, i]
                        cls_score.backward(retain_graph=True)
                        cam = cam_extractor(i, cls_out)[0].cpu().detach().numpy().squeeze()
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        cams_all.append(cam)
                    cams_fused = sum(cam * probs[i] for i, cam in enumerate(cams_all))
                    cams_fused_full = np.array(Image.fromarray((cams_fused * 255).astype(np.uint8)).resize(IMG_SIZE)) / 255.0
                    cam_color = (cm.get_cmap("jet")(cams_fused_full)[:, :, :3] * 255).astype(np.uint8)
                    overlay_cam = blend_overlay(base_arr, cam_color, alpha=0.5)

                    # Difference overlay
                    diff_overlay = base_arr.copy()
                    missed = np.logical_and(seg_mask_full > 0.5, student_mask_resized <= 0.5)
                    extra = np.logical_and(student_mask_resized > 0.5, seg_mask_full <= 0.5)
                    diff_overlay[missed] = (diff_overlay[missed] * 0.5 + np.array([255, 255, 0]) * 0.5).astype(np.uint8)
                    diff_overlay[extra] = (diff_overlay[extra] * 0.5 + np.array([255, 0, 255]) * 0.5).astype(np.uint8)

                    st.session_state.ai_results = {
                        "ai_class": ai_class,
                        "probs": probs.tolist(),
                        "dice": float(dice),
                        "overlay_student": overlay_student,
                        "overlay_ai": overlay_ai,
                        "overlay_cam": overlay_cam,
                        "diff_overlay": diff_overlay
                    }
                    st.session_state.revealed = True
                    st.success("✅ AI analysis complete! Scroll down to see the comparison.")

                except Exception as e:
                    st.error(f"❌ AI analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # ---------------------------
    if st.session_state.revealed and st.session_state.ai_results is not None:
        r = st.session_state.ai_results
        
        st.markdown("---")
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Step 4: Comparative Analysis & Learning")

        st.markdown("#### 🖼️ Visual Comparison")
        cols = st.columns(4)
        with cols[0]:
            st.image(r["overlay_student"], caption="👨‍🎓 Your Segmentation", use_column_width=True)
        with cols[1]:
            st.image(r["overlay_ai"], caption="🤖 AI Segmentation", use_column_width=True)
        with cols[2]:
            st.image(r["overlay_cam"], caption="🔥 AI Attention Map (Grad-CAM)", use_column_width=True)
        with cols[3]:
            st.image(r["diff_overlay"], caption="⚠️ Difference Analysis", use_column_width=True)
        
        st.markdown("*Yellow = Missed by you | Magenta = Extra area you included*")

        st.markdown("#### 📈 Performance Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("Dice Similarity Score", f"{r['dice']:.3f}")
            if r['dice'] >= 0.8:
                st.markdown("🟢 **Excellent** segmentation overlap!")
            elif r['dice'] >= 0.6:
                st.markdown("🟡 **Good** segmentation, minor refinements needed")
            else:
                st.markdown("🔴 **Needs improvement** - review lesion boundaries")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.metric("Your Classification", st.session_state.student_class.upper())
            st.metric("AI Classification", r['ai_class'].upper())
            if st.session_state.student_class == r['ai_class']:
                st.markdown("✅ **Match!** Your classification agrees with AI")
            else:
                st.markdown("⚠️ **Discordant** - Review key features")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
            st.markdown("**AI Confidence:**")
            for i, class_name in enumerate(CLASS_NAMES):
                confidence = r['probs'][i] * 100
                st.markdown(f"- {class_name.capitalize()}: {confidence:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 👨‍⚕️ Dr. Nova's Final Assessment")
        
        final_prompt = f"""You are Dr. Nova, an attending breast radiologist providing final feedback to a radiology resident after they've compared their work with AI analysis.

PERFORMANCE DATA:
- Dice Score (segmentation overlap): {r['dice']:.3f}
- Student Classification: {st.session_state.student_class}
- AI Classification: {r['ai_class']}
- AI Confidence: {', '.join([f'{CLASS_NAMES[i]}={r["probs"][i]*100:.1f}%' for i in range(len(CLASS_NAMES))])}
- Student's Reasoning: {st.session_state.student_reasoning if st.session_state.student_reasoning else "Not documented"}

Provide comprehensive educational feedback in this structure:

**1. Segmentation Performance:**
- Evaluate their Dice score and what it means
- Identify specific areas they missed or over-included
- Provide tips for more accurate lesion boundary identification

**2. Classification Assessment:**
- Evaluate their classification choice
- If correct: Reinforce their reasoning and key features they identified
- If incorrect: Explain what features they may have missed and why the correct classification is more appropriate

**3. Key Learning Points:**
- Highlight 2-3 specific ultrasound features they should focus on
- Relate findings to BI-RADS criteria
- Provide clinical context (what would you do next clinically?)

**4. Action Items for Improvement:**
- Give 2-3 specific, actionable recommendations for their next case
- Suggest areas of study or practice

Keep your tone supportive and educational. This is a learning experience, not a test."""

        final_reply = """**Dr. Nova's Assessment** (AI tutor unavailable - Groq not configured)

**Segmentation Performance:**
Your segmentation shows good effort in identifying the region of interest. Focus on including the entire lesion with clear margins while minimizing normal tissue.

**Classification Guidance:**
When classifying breast lesions, systematically evaluate:
- Shape and orientation (oval vs irregular, wider-than-tall vs taller-than-wide)
- Margin characteristics (circumscribed vs spiculated)
- Echo pattern (hypoechoic, isoechoic, hyperechoic)
- Posterior acoustic features (enhancement vs shadowing)

**Next Steps:**
1. Practice on more cases to develop pattern recognition
2. Review BI-RADS criteria for each classification
3. Document your reasoning systematically for each case
4. Compare your assessments with expert interpretations

Keep practicing! Radiology interpretation improves significantly with deliberate practice and feedback."""

        if groq_client:
            try:
                resp_final = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "You are Dr. Nova, an expert breast radiologist and compassionate medical educator who provides structured, actionable feedback to help residents improve."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=800
                )
                final_reply = resp_final.choices[0].message.content
            except Exception as e:
                final_reply += f"\n\n*(Error connecting to AI tutor: {str(e)})*"

        st.markdown(final_reply)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div class="success-box">
        <h4>🎯 Ready for More Practice?</h4>
        <p>Click the "Next Case" button in the top section to continue building your radiology interpretation skills!</p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 Please select a case from the sidebar to begin your radiology practice session.")
